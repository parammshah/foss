"""
Outcome Matcher for Reserve Juggler.
Matches predictions, decisions, and expected costs to subsequent observed or simulated execution outcomes.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
from reserve_juggler.integration.contracts import LearningEvent
from reserve_juggler.learning.models import MatchedOutcomePair
from reserve_juggler.learning.store import EventStore

logger = logging.getLogger(__name__)


class OutcomeMatcher:
    """
    Pairs decision and forecast events with subsequent execution outcome events
    using correlation IDs.
    """

    def __init__(self, event_store: EventStore):
        self.store = event_store

    def match_all(self) -> List[MatchedOutcomePair]:
        """
        Scans event store and produces matched pairs between DECISION_GENERATED
        and OUTCOME_OBSERVED events.
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

        pairs: List[MatchedOutcomePair] = []
        for corr_id, dec_ev in decisions.items():
            if corr_id in outcomes:
                out_ev = outcomes[corr_id]
                dec_p = dec_ev.payload
                out_p = out_ev.payload

                pred_cost = float(dec_p.get("estimated_costs", {}).get("total", 0.0))
                real_cost = float(out_p.get("realized_costs", {}).get("total", 0.0))
                cost_err = real_cost - pred_cost
                cost_err_pct = (cost_err / pred_cost * 100.0) if pred_cost > 0 else 0.0

                shortfall = float(out_p.get("liquidity_shortfall", 0.0))

                pair = MatchedOutcomePair(
                    correlation_id=corr_id,
                    decision_id=dec_p.get("decision_id", corr_id),
                    timestamp=dec_ev.timestamp,
                    predicted_outflow=float(dec_p.get("amount", 0.0)),
                    expected_cost=pred_cost,
                    selected_action=dec_p.get("selected_action", "UNKNOWN"),
                    target_asset=dec_p.get("target_asset", "UNKNOWN"),
                    model_version=dec_ev.associated_versions.get("decision_engine", "1.0.0"),
                    realized_amount=float(out_p.get("realized_amount", 0.0)),
                    realized_cost=real_cost,
                    liquidity_shortfall=shortfall,
                    settlement_status=out_p.get("settlement_status", "SETTLED"),
                    cost_error=cost_err,
                    cost_error_pct=round(cost_err_pct, 2),
                    is_shortfall=(shortfall > 0.0),
                )
                pairs.append(pair)

        return pairs
