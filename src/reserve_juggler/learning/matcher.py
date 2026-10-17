"""
Outcome Matcher for Reserve Juggler.
Separates forecast matching from financial decision evaluation:
1. match_forecasts(): Pairs ForecastPoints to actual observed customer cash flows across horizons.
2. match_decisions(): Pairs DecisionRecords to actual execution outcomes.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from reserve_juggler.contracts import LearningEvent
from reserve_juggler.learning.models import (
    MatchedDecisionOutcome,
    MatchedForecastOutcome,
)
from reserve_juggler.learning.store import EventStore

logger = logging.getLogger(__name__)


class OutcomeMatcher:
    """
    Pairs forecast and decision events with subsequent execution and cash flow observations.
    """

    def __init__(self, event_store: EventStore):
        self.store = event_store

    def match_forecasts(self) -> List[MatchedForecastOutcome]:
        """
        Matches FORECAST_INFERENCE_RECORDED events with corresponding
        ACTUAL_FLOWS_OBSERVED / CUSTOMER_CASH_FLOW events matching target timestamp and horizon.
        Never matches a forecast to an action amount.
        """
        all_events = self.store.get_all()
        forecast_events: List[LearningEvent] = []
        actual_flow_events: Dict[str, Dict[str, Any]] = {}  # target_timestamp or corr_id -> payload
        used_actual_event_ids: set = set()

        for ev in all_events:
            if ev.versioned_event_type == "FORECAST_INFERENCE_RECORDED":
                forecast_events.append(ev)
            elif ev.versioned_event_type in ("ACTUAL_FLOWS_OBSERVED", "ACTUAL_CASH_FLOW_OBSERVED", "CUSTOMER_CASH_FLOW", "REVEALED_CASH_FLOWS"):
                p = dict(ev.payload)
                p["_event_id"] = ev.event_id
                p["_event_timestamp"] = ev.timestamp
                ts = p.get("target_timestamp") or ev.timestamp
                actual_flow_events[ts] = p
                if ev.correlation_id:
                    actual_flow_events[ev.correlation_id] = p

        matched: List[MatchedForecastOutcome] = []

        for f_ev in forecast_events:
            corr_id = f_ev.correlation_id or f_ev.event_id
            payload = f_ev.payload
            points = payload.get("point_forecasts", {})
            model_ver = f_ev.associated_versions.get("forecasting_model", "v1.4.0")

            for horizon, pt_data in points.items():
                pred_net = float(pt_data.get("expected_net_flow", 0.0))
                pred_in = float(pt_data.get("expected_inflow", 0.0))
                pred_out = float(pt_data.get("expected_outflow", 0.0))

                # Part C2: Do NOT fabricate uncertainty intervals with arbitrary million dollar offsets!
                p05 = float(pt_data["uncertainty_p05"]) if pt_data.get("uncertainty_p05") is not None else None
                p50 = float(pt_data["uncertainty_p50"]) if pt_data.get("uncertainty_p50") is not None else None
                p90 = float(pt_data["uncertainty_p90"]) if pt_data.get("uncertainty_p90") is not None else None
                p95 = float(pt_data["uncertainty_p95"]) if pt_data.get("uncertainty_p95") is not None else None
                p99 = float(pt_data["uncertainty_p99"]) if pt_data.get("uncertainty_p99") is not None else None

                target_ts = pt_data.get("target_timestamp") or payload.get("target_timestamp")
                
                # If target_ts is missing, allow unambiguous correlation ID match to an actual flow event,
                # but NEVER silently replace target_ts with forecast creation time (f_ev.timestamp).
                actual_data = None
                if target_ts:
                    cand = actual_flow_events.get(target_ts)
                    if cand and cand.get("_event_id") not in used_actual_event_ids:
                        actual_data = cand
                elif corr_id in actual_flow_events:
                    cand = actual_flow_events[corr_id]
                    if cand and cand.get("_event_id") not in used_actual_event_ids:
                        target_ts = cand.get("target_timestamp") or cand.get("_event_timestamp")
                        actual_data = cand

                if not target_ts:
                    # Part C1: Never silently replace missing target timestamp with forecast creation timestamp!
                    matched.append(
                        MatchedForecastOutcome(
                            correlation_id=corr_id,
                            forecast_id=f"{corr_id}_{horizon}",
                            target_timestamp="",
                            horizon=horizon,
                            predicted_net_flow=float(pt_data.get("expected_net_flow", 0.0)),
                            predicted_inflow=float(pt_data.get("expected_inflow", 0.0)),
                            predicted_outflow=float(pt_data.get("expected_outflow", 0.0)),
                            uncertainty_p05=None,
                            uncertainty_p50=None,
                            uncertainty_p90=None,
                            uncertainty_p95=None,
                            uncertainty_p99=None,
                            model_version=model_ver,
                            actual_net_flow=None,
                            is_eligible=False,
                            ineligibility_reason="Missing forecast target timestamp; target timestamp cannot default to forecast creation time.",
                        )
                    )
                    continue

                if actual_data and actual_data.get("_event_id"):
                    used_actual_event_ids.add(actual_data["_event_id"])

                if actual_data:
                    # Determine actual net flow (Part C2: never default missing to 0.0)
                    act_net = None
                    if actual_data.get("actual_net_flow") is not None:
                        act_net = float(actual_data["actual_net_flow"])
                    elif actual_data.get("net_flow") is not None:
                        act_net = float(actual_data["net_flow"])
                    elif actual_data.get("actual_inflow") is not None and actual_data.get("actual_outflow") is not None:
                        act_net = float(actual_data["actual_inflow"]) - float(actual_data["actual_outflow"])

                    act_in = float(actual_data["actual_inflow"]) if actual_data.get("actual_inflow") is not None else (float(actual_data["inflow"]) if actual_data.get("inflow") is not None else None)
                    act_out = float(actual_data["actual_outflow"]) if actual_data.get("actual_outflow") is not None else (float(actual_data["outflow"]) if actual_data.get("outflow") is not None else None)

                    if act_net is not None:
                        abs_err = abs(act_net - pred_net)
                        # Empirical interval coverage only if valid stored bounds exist
                        in_90 = None
                        if p05 is not None and p95 is not None:
                            in_90 = (act_net >= p05) and (act_net <= p95)

                        matched.append(
                            MatchedForecastOutcome(
                                correlation_id=corr_id,
                                forecast_id=f"{corr_id}_{horizon}",
                                target_timestamp=target_ts,
                                horizon=horizon,
                                predicted_net_flow=pred_net,
                                predicted_inflow=pred_in,
                                predicted_outflow=pred_out,
                                uncertainty_p05=p05,
                                uncertainty_p50=p50,
                                uncertainty_p90=p90,
                                uncertainty_p95=p95,
                                uncertainty_p99=p99,
                                model_version=model_ver,
                                actual_net_flow=act_net,
                                actual_inflow=act_in,
                                actual_outflow=act_out,
                                is_eligible=True,
                                ineligibility_reason=None,
                                abs_error=abs_err,
                                within_90_interval=in_90,
                            )
                        )
                    else:
                        # Actual event was present but missing actual cash flow observations (never convert missing to 0.0)
                        matched.append(
                            MatchedForecastOutcome(
                                correlation_id=corr_id,
                                forecast_id=f"{corr_id}_{horizon}",
                                target_timestamp=target_ts,
                                horizon=horizon,
                                predicted_net_flow=pred_net,
                                predicted_inflow=pred_in,
                                predicted_outflow=pred_out,
                                uncertainty_p05=p05,
                                uncertainty_p50=p50,
                                uncertainty_p90=p90,
                                uncertainty_p95=p95,
                                uncertainty_p99=p99,
                                model_version=model_ver,
                                actual_net_flow=None,
                                actual_inflow=act_in,
                                actual_outflow=act_out,
                                is_eligible=False,
                                ineligibility_reason="Missing actual observed cash flows in ledger event. Missing data cannot be converted to 0.0.",
                            )
                        )
                else:
                    # Stored forecast without observed actual yet
                    matched.append(
                        MatchedForecastOutcome(
                            correlation_id=corr_id,
                            forecast_id=f"{corr_id}_{horizon}",
                            target_timestamp=target_ts,
                            horizon=horizon,
                            predicted_net_flow=pred_net,
                            predicted_inflow=pred_in,
                            predicted_outflow=pred_out,
                            uncertainty_p05=p05,
                            uncertainty_p50=p50,
                            uncertainty_p90=p90,
                            uncertainty_p95=p95,
                            uncertainty_p99=p99,
                            model_version=model_ver,
                            actual_net_flow=None,
                            is_eligible=False,
                            ineligibility_reason="Target horizon outcome not yet observed or pending settlement.",
                        )
                    )

        return matched

    def match_decisions(self) -> List[MatchedDecisionOutcome]:
        """
        Matches DECISION_GENERATED events with OUTCOME_OBSERVED / EXECUTION_OUTCOME events.
        Distinguishes approved/executed actions from cancelled, rejected, and pending proposals.
        """
        all_events = self.store.get_all()
        decisions: Dict[str, LearningEvent] = {}
        outcomes: Dict[str, LearningEvent] = {}

        for ev in all_events:
            corr_id = ev.correlation_id
            if not corr_id:
                continue
            if ev.versioned_event_type == "DECISION_GENERATED":
                decisions[corr_id] = ev
            elif ev.versioned_event_type in ("OUTCOME_OBSERVED", "EXECUTION_OUTCOME"):
                outcomes[corr_id] = ev

        matched: List[MatchedDecisionOutcome] = []

        for corr_id, dec_ev in decisions.items():
            dec_p = dec_ev.payload
            out_ev = outcomes.get(corr_id)
            
            pred_cost = float(dec_p.get("estimated_costs", {}).get("total", 0.0))
            action = dec_p.get("selected_action", "UNKNOWN")
            asset = dec_p.get("target_asset", "UNKNOWN")
            amt = float(dec_p.get("amount", 0.0))
            gates = dec_p.get("gate_outcomes", [])

            if out_ev:
                out_p = out_ev.payload
                raw_status = out_p.get("settlement_status", "UNKNOWN").upper()
                real_cost = float(out_p.get("realized_costs", {}).get("total", 0.0))
                shortfall = float(out_p.get("liquidity_shortfall", 0.0))
                
                # Execution status categorization (Part C3)
                is_exec = raw_status in ("SETTLED", "EXECUTED", "CONFIRMED")
                is_cancelled_rejected = raw_status in ("CANCELLED", "REJECTED")
                is_pending = raw_status in ("PENDING", "DELAYED", "EXPIRED", "UNRESOLVED", "UNKNOWN")

                if is_exec:
                    cost_err = real_cost - pred_cost
                    cost_err_bps = (cost_err / pred_cost * 10000.0) if pred_cost > 0 else 0.0
                    app_status = out_p.get("approval_status", "APPROVED")
                elif is_cancelled_rejected:
                    real_cost = 0.0
                    cost_err = 0.0
                    cost_err_bps = 0.0
                    app_status = out_p.get("approval_status", "CANCELLED" if raw_status == "CANCELLED" else "REJECTED")
                else:
                    # Pending, delayed, or unresolved
                    real_cost = 0.0
                    cost_err = 0.0
                    cost_err_bps = 0.0
                    app_status = "PENDING_APPROVAL"

                matched.append(
                    MatchedDecisionOutcome(
                        correlation_id=corr_id,
                        decision_id=dec_p.get("decision_id", corr_id),
                        timestamp=dec_ev.timestamp,
                        selected_action=action,
                        target_asset=asset,
                        proposed_amount=amt,
                        approval_status=app_status,
                        is_executed=is_exec,
                        execution_status=raw_status,
                        expected_cost=pred_cost,
                        realized_cost=real_cost,
                        cost_error=cost_err,
                        cost_error_bps=round(cost_err_bps, 1),
                        liquidity_shortfall=shortfall if is_exec else 0.0,
                        is_shortfall=(shortfall > 0.0 and is_exec),
                        hard_gates=gates,
                    )
                )
            else:
                # Decision proposed/pending without outcome yet
                matched.append(
                    MatchedDecisionOutcome(
                        correlation_id=corr_id,
                        decision_id=dec_p.get("decision_id", corr_id),
                        timestamp=dec_ev.timestamp,
                        selected_action=action,
                        target_asset=asset,
                        proposed_amount=amt,
                        approval_status="PENDING_APPROVAL",
                        is_executed=False,
                        execution_status="PENDING",
                        expected_cost=pred_cost,
                        realized_cost=0.0,
                        cost_error=0.0,
                        cost_error_bps=0.0,
                        liquidity_shortfall=0.0,
                        is_shortfall=False,
                        hard_gates=gates,
                    )
                )

        return matched

    # Backward compatibility
    def match_all(self) -> List[MatchedDecisionOutcome]:
        return self.match_decisions()
