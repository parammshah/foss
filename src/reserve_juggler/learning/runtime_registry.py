"""
Runtime Component Registry for Reserve Juggler.
Maintains live, active parameter providers and runtime state for promotable components,
ensuring that model promotion and rollback directly update operational execution behavior
rather than merely changing metadata strings.
"""

from __future__ import annotations
import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RuntimeProvider(ABC):
    """
    Abstract interface for operational treasury subsystem consumers.
    Promoting a candidate must mutate the underlying provider.
    """

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Name of subsystem component (e.g. FORECASTING, SENTINEL_RISK, DECISION_OPTIMIZER)."""
        pass

    @abstractmethod
    def get_parameter(self, parameter_key: str, default: Any = None) -> Any:
        """Retrieves active runtime parameter from the operational consumer."""
        pass

    @abstractmethod
    def set_parameter(self, parameter_key: str, new_value: Any) -> Tuple[bool, str]:
        """Mutates active operational consumer behavior with the new parameter."""
        pass


class SentinelRuntimeProvider(RuntimeProvider):
    """Binds live parameter updates directly to MacroPanicSentinel and EventRiskInterpreter."""

    def __init__(self, sentinel: Any):
        self.sentinel = sentinel

    @property
    def consumer(self) -> Any:
        return self.sentinel

    @property
    def component_name(self) -> str:
        return "SENTINEL_RISK"

    def get_parameter(self, parameter_key: str, default: Any = None) -> Any:
        if parameter_key == "high_severity_threshold":
            return getattr(self.sentinel.interpreter, "high_severity_threshold", default)
        elif parameter_key == "critical_severity_threshold":
            return getattr(self.sentinel.interpreter, "critical_severity_threshold", default)
        elif parameter_key == "stale_hours_threshold":
            return getattr(self.sentinel.interpreter, "stale_hours_threshold", default)
        elif parameter_key == "panic_sentinel_alert_threshold":
            return getattr(self.sentinel, "alert_threshold", 60.0)
        return default

    def set_parameter(self, parameter_key: str, new_value: Any) -> Tuple[bool, str]:
        try:
            val = float(new_value)
            if parameter_key == "high_severity_threshold":
                self.sentinel.interpreter.high_severity_threshold = val
            elif parameter_key == "critical_severity_threshold":
                self.sentinel.interpreter.critical_severity_threshold = val
            elif parameter_key == "stale_hours_threshold":
                self.sentinel.interpreter.stale_hours_threshold = val
            elif parameter_key == "panic_sentinel_alert_threshold":
                self.sentinel.alert_threshold = val
            else:
                return (False, f"Unknown parameter '{parameter_key}' for SentinelRuntimeProvider.")
            return (True, f"Applied {parameter_key}={val} to active MacroPanicSentinel.")
        except Exception as e:
            return (False, f"Failed to apply {parameter_key} to Sentinel: {e}")


class GenericRuntimeProvider(RuntimeProvider):
    """Configurable typed runtime provider for connected treasury adapters or models."""

    def __init__(self, name: str, initial_params: Optional[Dict[str, Any]] = None):
        self._name = name.upper()
        self._params: Dict[str, Any] = initial_params or {}

    @property
    def component_name(self) -> str:
        return self._name

    def get_parameter(self, parameter_key: str, default: Any = None) -> Any:
        return self._params.get(parameter_key, default)

    def set_parameter(self, parameter_key: str, new_value: Any) -> Tuple[bool, str]:
        self._params[parameter_key] = new_value
        return (True, f"Applied {parameter_key}={new_value} to active {self._name}.")


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

        # Active connected provider instances (empty by default; real consumers must register explicitly)
        self._providers: Dict[str, RuntimeProvider] = {}

        # Version-specific runtime configuration history (Part B2, B3)
        self._version_configs: Dict[str, Dict[str, Dict[str, Any]]] = {
            k: {"v1.0.0": copy.deepcopy(v), "v1.4.0": copy.deepcopy(v), "v1.2.0": copy.deepcopy(v), "v2.0.0": copy.deepcopy(v)}
            for k, v in self._runtime_configs.items()
        }

        # Snapshot stack for rollback
        self._history_snapshots: Dict[str, list[Dict[str, Any]]] = {
            k: [copy.deepcopy(v)] for k, v in self._runtime_configs.items()
        }

    def record_promoted_version(self, component: str, version_tag: str) -> None:
        """Durably associates the active runtime configuration with a promoted version."""
        comp = component.upper()
        self._version_configs.setdefault(comp, {})[version_tag] = copy.deepcopy(self._runtime_configs.get(comp, {}))

    def has_version_config(self, component: str, version_tag: str) -> bool:
        """Returns True if a specific version has a restorable configuration."""
        comp = component.upper()
        return comp in self._version_configs and version_tag in self._version_configs[comp]

    def get_component_status(self, component: str) -> str:
        """Returns the truthful integration status of a subsystem."""
        comp = component.upper()
        if comp in self._providers:
            return "CONNECTED"
        if comp in self._runtime_configs:
            return "UNINTEGRATED"
        return "UNAVAILABLE"

    def register_provider(self, component: str, provider: RuntimeProvider) -> None:
        """Registers an active operational component consumer."""
        comp = component.upper()
        self._providers[comp] = provider
        if hasattr(provider, "_params"):
            for k, v in provider._params.items():
                self._runtime_configs.setdefault(comp, {})[k] = copy.deepcopy(v)
        # Synchronize configuration with provider
        for k, v in self._runtime_configs.get(comp, {}).items():
            provider.set_parameter(k, v)
        self._history_snapshots[comp] = [copy.deepcopy(self._runtime_configs.get(comp, {}))]
        if comp in self._version_configs and "v1.0.0" in self._version_configs[comp]:
            self._version_configs[comp]["v1.0.0"] = copy.deepcopy(self._runtime_configs.get(comp, {}))
        logger.info("Registered active runtime provider for subsystem %s.", comp)

    def unregister_provider(self, component: str) -> None:
        """Unregisters a provider, making the component unintegrated."""
        self._providers.pop(component.upper(), None)

    def has_active_provider(self, component: str) -> bool:
        """Returns True if the subsystem has an active registered consumer."""
        return component.upper() in self._providers

    def get_provider(self, component: str) -> Optional[Any]:
        """Returns the registered provider for a component, if any."""
        return self._providers.get(component.upper())

    def get_runtime_config(self, component: str) -> Dict[str, Any]:
        """Returns a copy of the active runtime configuration for a component."""
        comp = component.upper()
        cfg = copy.deepcopy(self._runtime_configs.get(comp, {}))
        # If provider is connected, merge provider's live values
        if comp in self._providers:
            provider = self._providers[comp]
            for k in cfg:
                live_v = provider.get_parameter(k)
                if live_v is not None:
                    cfg[k] = live_v
        return cfg

    def get_parameter(self, component: str, parameter_key: str, default: Any = None) -> Any:
        """Retrieves a single live runtime parameter value."""
        comp = component.upper()
        if comp in self._providers:
            val = self._providers[comp].get_parameter(parameter_key)
            if val is not None:
                return val
        comp_cfg = self._runtime_configs.get(comp, {})
        return comp_cfg.get(parameter_key, default)

    get_runtime_parameter = get_parameter

    def apply_runtime_update(
        self,
        component: str,
        parameter_key: str,
        new_value: Any,
        require_active_provider: bool = False,
    ) -> Tuple[bool, str, Any]:
        """
        Updates an operational runtime parameter upon authorized candidate promotion.
        If require_active_provider is True and no consumer is registered, blocks promotion.
        Returns (success, message, old_value).
        """
        comp = component.upper()
        if comp not in self._runtime_configs:
            return (False, f"Component '{comp}' is not a recognized subsystem.", None)

        if require_active_provider and comp not in self._providers:
            return (
                False,
                f"Component '{comp}' does not have an active registered runtime consumer. "
                f"Cannot execute live promotion on an UNINTEGRATED component.",
                None,
            )

        old_val = self.get_parameter(comp, parameter_key)

        # Snapshot current state before update
        self._history_snapshots[comp].append(copy.deepcopy(self._runtime_configs[comp]))

        # Apply live update to registered provider if available
        if comp in self._providers:
            ok, p_msg = self._providers[comp].set_parameter(parameter_key, new_value)
            if not ok:
                self._history_snapshots[comp].pop()
                return (False, f"Runtime provider rejected parameter update: {p_msg}", old_val)

        # Apply to configuration store
        self._runtime_configs[comp][parameter_key] = new_value
        logger.info(
            "Runtime parameter updated for %s: %s = %s (was %s)", comp, parameter_key, new_value, old_val
        )
        return (True, f"Runtime parameter '{parameter_key}' updated to {new_value}.", old_val)

    def restore_runtime_state(self, component: str, target_version: Optional[str] = None) -> Tuple[bool, str]:
        """
        Restores the operational runtime configuration on rollback (Part B3, E4).
        If target_version is provided, restores the configuration durably associated with that version.
        Otherwise restores the immediately prior snapshot.
        """
        comp = component.upper()
        if target_version:
            if comp not in self._version_configs or target_version not in self._version_configs[comp]:
                return (False, f"Target version '{target_version}' has no valid restorable runtime configuration for {comp}.")
            target_snapshot = self._version_configs[comp][target_version]
        else:
            if comp not in self._history_snapshots or len(self._history_snapshots[comp]) <= 1:
                return (False, f"No previous runtime snapshot available to restore for {comp}.")
            self._history_snapshots[comp].pop()
            target_snapshot = self._history_snapshots[comp][-1]

        # Restore on live provider if registered
        if comp in self._providers:
            for k, v in target_snapshot.items():
                ok, p_msg = self._providers[comp].set_parameter(k, v)
                if not ok:
                    return (False, f"Live runtime consumer failed to apply parameter '{k}' on rollback: {p_msg}")

        self._runtime_configs[comp] = copy.deepcopy(target_snapshot)
        version_str = f"to version {target_version}" if target_version else "to previous snapshot"
        logger.info("Runtime state restored for %s %s.", comp, version_str)
        return (True, f"Runtime configuration restored for {comp} {version_str}.")
