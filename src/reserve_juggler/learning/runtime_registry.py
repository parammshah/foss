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

    def __init__(self, persistence_store: Optional[Any] = None):
        self.persistence_store = persistence_store
        self._store = getattr(persistence_store, "sqlite_store", None) if persistence_store else None
        if self._store is None and hasattr(persistence_store, "save_runtime_version_config"):
            self._store = persistence_store

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
        # Baseline version tags aligned with VersionRegistry
        self._initial_versions: Dict[str, str] = {
            "FORECASTING": "v1.4.0",
            "DECISION_OPTIMIZER": "v1.2.0",
            "SENTINEL_RISK": "v1.0.0",
            "SCENARIO_GENERATOR": "v1.1.0",
            "EXPLAINABILITY": "v1.0.0",
        }

        # Active connected provider instances (empty by default; real consumers must register explicitly)
        self._providers: Dict[str, RuntimeProvider] = {}

        # Version-specific runtime configuration history (Part B2, B3)
        self._version_configs: Dict[str, Dict[str, Dict[str, Any]]] = {
            k: {} for k in self._runtime_configs
        }

        # Track which components were recovered from durable store
        self._durable_loaded_components: set = set()

        # Recover from durable persistence if available
        if self._store:
            try:
                all_saved = self._store.get_all_runtime_configs()
                for comp, v_map in all_saved.items():
                    self._version_configs.setdefault(comp, {}).update(v_map)

                active_versions = self._store.get_active_runtime_versions()
                for comp, (ver_tag, active_cfg) in active_versions.items():
                    self._runtime_configs[comp] = copy.deepcopy(active_cfg)
                    self._durable_loaded_components.add(comp)
                    logger.info("Restored active runtime configuration for %s version %s from persistent store.", comp, ver_tag)

                # Persist baseline configurations for subsystems that have no records in store yet
                for comp, cfg in self._runtime_configs.items():
                    base_ver = self._initial_versions.get(comp, "v1.0.0")
                    if comp not in all_saved or not all_saved[comp]:
                        self._store.save_runtime_version_config(comp, base_ver, copy.deepcopy(cfg), is_active=True)
                        self._version_configs.setdefault(comp, {})[base_ver] = copy.deepcopy(cfg)
            except Exception as e:
                logger.error("Failed to recover runtime configs from persistent store: %s", e)
        else:
            for comp, cfg in self._runtime_configs.items():
                base_ver = self._initial_versions.get(comp, "v1.0.0")
                self._version_configs.setdefault(comp, {})[base_ver] = copy.deepcopy(cfg)

        # Snapshot stack for rollback
        self._history_snapshots: Dict[str, list[Dict[str, Any]]] = {
            k: [copy.deepcopy(v)] for k, v in self._runtime_configs.items()
        }

        # Track components requiring recovery due to unrecoverable state/compensation failures
        self._recovery_required: Dict[str, str] = {}

    def mark_recovery_required(self, component: str, reason: str) -> None:
        """Marks a component as requiring manual recovery, blocking any further promotion."""
        comp = component.upper()
        self._recovery_required[comp] = reason
        logger.critical("Subsystem %s marked RECOVERY_REQUIRED: %s", comp, reason)

    def is_recovery_required(self, component: str) -> bool:
        """Returns True if the component has had an unrecoverable state failure."""
        return component.upper() in self._recovery_required

    def get_recovery_reason(self, component: str) -> Optional[str]:
        """Returns the failure reason if the component requires recovery."""
        return self._recovery_required.get(component.upper())

    def clear_recovery_required(self, component: str) -> None:
        """Clears the recovery-required flag after repair."""
        self._recovery_required.pop(component.upper(), None)

    def record_promoted_version(self, component: str, version_tag: str) -> None:
        """Durably associates the active runtime configuration with a promoted version."""
        comp = component.upper()
        cfg = copy.deepcopy(self._runtime_configs.get(comp, {}))
        self._version_configs.setdefault(comp, {})[version_tag] = cfg
        if self._store:
            try:
                self._store.save_runtime_version_config(comp, version_tag, cfg, is_active=True)
            except Exception as e:
                logger.error("Failed to persist promoted version config for %s %s: %s", comp, version_tag, e)
                raise

    def has_version_config(self, component: str, version_tag: str) -> bool:
        """Returns True if a specific version has a restorable configuration."""
        comp = component.upper()
        if self._store:
            try:
                configs = self._store.get_runtime_version_configs(comp)
                return version_tag in configs
            except Exception as e:
                logger.error("Failed to query runtime configs from store for %s: %s", comp, e)
                return False
        return comp in self._version_configs and version_tag in self._version_configs[comp]

    def get_component_status(self, component: str) -> str:
        """Returns the truthful integration status of a subsystem."""
        comp = component.upper()
        if self.is_recovery_required(comp):
            return "RECOVERY_REQUIRED"
        if comp in self._providers:
            return "CONNECTED"
        if comp in self._runtime_configs:
            return "UNINTEGRATED"
        return "UNAVAILABLE"

    def register_provider(self, component: str, provider: RuntimeProvider) -> None:
        """Registers an active operational component consumer."""
        comp = component.upper()
        self._providers[comp] = provider

        # If component was NOT restored from durable storage, initialize from provider defaults
        if comp not in self._durable_loaded_components:
            if hasattr(provider, "_params"):
                for k, v in provider._params.items():
                    self._runtime_configs.setdefault(comp, {})[k] = copy.deepcopy(v)
            base_ver = self._initial_versions.get(comp, "v1.0.0")
            if comp in self._version_configs:
                self._version_configs[comp][base_ver] = copy.deepcopy(self._runtime_configs.get(comp, {}))
            if self._store:
                try:
                    active_map = self._store.get_active_runtime_versions()
                    if comp not in active_map or active_map[comp][0] == base_ver:
                        self._store.save_runtime_version_config(comp, base_ver, self._runtime_configs[comp], is_active=True)
                except Exception as e:
                    logger.debug("Could not sync provider params to store: %s", e)

        # Always synchronize configuration from active registry to provider
        for k, v in self._runtime_configs.get(comp, {}).items():
            provider.set_parameter(k, v)

        self._history_snapshots[comp] = [copy.deepcopy(self._runtime_configs.get(comp, {}))]
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
        if self.is_recovery_required(comp):
            return (
                False,
                f"Component '{comp}' is in RECOVERY_REQUIRED state ({self.get_recovery_reason(comp)}). "
                f"Cannot apply runtime updates until repaired.",
                None,
            )

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
        Restores the operational runtime configuration on rollback.
        If target_version is provided, restores the configuration durably associated with that version.
        Otherwise restores the immediately prior snapshot.
        Maintains strict transactional consistency: if persistent store write fails, reverts in-memory
        and live provider state, returning failure.
        """
        comp = component.upper()
        original_snapshot = copy.deepcopy(self._runtime_configs.get(comp, {}))

        if target_version:
            if self._store:
                try:
                    store_configs = self._store.get_runtime_version_configs(comp)
                except Exception as e:
                    return (False, f"Failed to query persistent database for {comp}: {e}")
                if target_version not in store_configs:
                    return (
                        False,
                        f"Target version '{target_version}' has no valid restorable runtime configuration for {comp}.",
                    )
                target_snapshot = store_configs[target_version]
                self._version_configs.setdefault(comp, {})[target_version] = copy.deepcopy(target_snapshot)
            else:
                if comp not in self._version_configs or target_version not in self._version_configs[comp]:
                    return (False, f"Target version '{target_version}' has no valid restorable runtime configuration for {comp}.")
                target_snapshot = self._version_configs[comp][target_version]
        else:
            if comp not in self._history_snapshots or len(self._history_snapshots[comp]) <= 1:
                return (False, f"No previous runtime snapshot available to restore for {comp}.")
            self._history_snapshots[comp].pop()
            target_snapshot = self._history_snapshots[comp][-1]

        # Snapshot original provider params for safe compensation
        original_provider_params = None
        if comp in self._providers and hasattr(self._providers[comp], "_params") and isinstance(self._providers[comp]._params, dict):
            original_provider_params = copy.deepcopy(self._providers[comp]._params)

        # 1. Restore on live provider if registered
        applied_keys = []
        if comp in self._providers:
            provider = self._providers[comp]
            # Synchronize provider params: discard old parameters that do not exist in target historical version
            if hasattr(provider, "_params") and isinstance(provider._params, dict):
                for old_k in list(provider._params.keys()):
                    if old_k not in target_snapshot:
                        del provider._params[old_k]

            for k, v in target_snapshot.items():
                ok, p_msg = provider.set_parameter(k, v)
                if not ok:
                    # Compensate: revert previously applied parameters back to original
                    if original_provider_params is not None and hasattr(provider, "_params"):
                        provider._params = copy.deepcopy(original_provider_params)
                    comp_failed = False
                    for rev_k, rev_v in original_snapshot.items():
                        ok_rev, _ = provider.set_parameter(rev_k, rev_v)
                        if not ok_rev:
                            comp_failed = True
                    if comp_failed:
                        self.mark_recovery_required(
                            comp,
                            f"Live runtime provider compensation failed after failure applying '{k}': {p_msg}",
                        )
                    return (False, f"Live runtime consumer failed to apply parameter '{k}' on rollback: {p_msg}")
                applied_keys.append(k)

        # 2. Update in-memory configuration
        self._runtime_configs[comp] = copy.deepcopy(target_snapshot)

        # 3. Transactionally persist active version to durable store
        if target_version and self._store:
            try:
                self._store.set_active_runtime_version(comp, target_version)
            except Exception as e:
                # PERSISTENCE FAILED: Must NOT report successful rollback!
                # Revert in-memory configuration back to original
                self._runtime_configs[comp] = copy.deepcopy(original_snapshot)

                # Revert live provider back to original
                if original_provider_params is not None and hasattr(self._providers[comp], "_params"):
                    self._providers[comp]._params = copy.deepcopy(original_provider_params)
                revert_failed = False
                if comp in self._providers:
                    for k, v in original_snapshot.items():
                        ok_p, p_msg = self._providers[comp].set_parameter(k, v)
                        if not ok_p:
                            revert_failed = True

                if revert_failed:
                    self.mark_recovery_required(
                        comp,
                        f"Failed to revert live provider after active-version persistence failure: {e}",
                    )

                logger.error(
                    "Persistent store failed to activate version %s on rollback: %s. State compensated.",
                    target_version,
                    e,
                )
                return (False, f"Persistent store failed to activate version '{target_version}' on rollback: {e}")

        version_str = f"to version {target_version}" if target_version else "to previous snapshot"
        logger.info("Runtime state restored for %s %s.", comp, version_str)
        return (True, f"Runtime configuration restored for {comp} {version_str}.")
