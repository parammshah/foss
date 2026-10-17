"""
Runtime Component Registry for Reserve Juggler.
Maintains live, active parameter providers and runtime state for promotable components,
ensuring that model promotion and rollback directly update operational execution behavior
rather than merely changing metadata strings.
"""

from __future__ import annotations
import copy
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RuntimeComponentRegistry:
    """
    Manages live parameter states for operational treasury subsystems.
    Enforces that candidate promotion directly mutates active runtime configuration,
    and rollback restores prior runtime parameters.
    """

    def __init__(self):
        # Operational runtime configurations
        self._runtime_configs: Dict[str, Dict[str, Any]] = {
            "FORECASTING": {
                "ensemble_combining_weights": {"lstm_weight": 0.50, "arimax_weight": 0.50},
                "arimax_lag_order": 3,
                "confidence_calibration_window_hours": 168,
            },
            "SENTINEL_RISK": {
                "panic_sentinel_alert_threshold": 60.0,
                "high_severity_threshold": 0.60,
                "critical_severity_threshold": 0.80,
                "stale_hours_threshold": 48.0,
            },
            "DECISION_OPTIMIZER": {
                "liquidity_buffer_multiplier": 1.25,
                "risk_penalty_weight": 0.35,
                "max_draw_per_tranche": 25_000_000.0,
            },
            "SCENARIO_GENERATOR": {
                "volatility_stress_scenario_multiplier": 1.5,
                "deposit_run_stress_percentage": 50.0,
            },
        }

        # Snapshot stack for rollback
        self._history_snapshots: Dict[str, list[Dict[str, Any]]] = {
            k: [copy.deepcopy(v)] for k, v in self._runtime_configs.items()
        }

    def get_runtime_config(self, component: str) -> Dict[str, Any]:
        """Returns a copy of the active runtime configuration for a component."""
        return copy.deepcopy(self._runtime_configs.get(component.upper(), {}))

    def get_parameter(self, component: str, parameter_key: str, default: Any = None) -> Any:
        """Retrieves a single live runtime parameter value."""
        comp_cfg = self._runtime_configs.get(component.upper(), {})
        return comp_cfg.get(parameter_key, default)

    get_runtime_parameter = get_parameter

    def apply_runtime_update(
        self, component: str, parameter_key: str, new_value: Any
    ) -> Tuple[bool, str, Any]:
        """
        Updates an operational runtime parameter upon authorized candidate promotion.
        Returns (success, message, old_value).
        """
        comp = component.upper()
        if comp not in self._runtime_configs:
            return (False, f"Component '{comp}' does not have an active runtime provider.", None)

        old_val = self._runtime_configs[comp].get(parameter_key)
        
        # Snapshot current state before update
        self._history_snapshots[comp].append(copy.deepcopy(self._runtime_configs[comp]))
        
        # Apply live update
        self._runtime_configs[comp][parameter_key] = new_value
        logger.info(
            "Runtime parameter updated for %s: %s = %s (was %s)", comp, parameter_key, new_value, old_val
        )
        return (True, f"Runtime parameter '{parameter_key}' updated to {new_value}.", old_val)

    def restore_runtime_state(self, component: str) -> Tuple[bool, str]:
        """Restores the previous operational runtime configuration on rollback."""
        comp = component.upper()
        if comp not in self._history_snapshots or len(self._history_snapshots[comp]) <= 1:
            return (False, f"No previous runtime snapshot available to restore for {comp}.")

        # Pop current, restore previous
        self._history_snapshots[comp].pop()
        prev_snapshot = self._history_snapshots[comp][-1]
        self._runtime_configs[comp] = copy.deepcopy(prev_snapshot)
        logger.info("Runtime state restored for %s to previous snapshot.", comp)
        return (True, f"Runtime configuration restored for {comp}.")
