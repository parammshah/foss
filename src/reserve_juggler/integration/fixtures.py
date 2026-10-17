"""
High-fidelity test fixtures and scenario generators representing Team 1 (Forecasting/Ledger)
and Team 2 (Decision/Risk/Optimization) contracts.
Used for local testing, demo scenarios, and interface isolation.
"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from reserve_juggler.integration.contracts import (
    AssetPosition,
    AlternativeAction,
    DecisionRecord,
    ForecastBundle,
    HardGateOutcome,
    HorizonForecast,
    Obligation,
    TreasuryState,
)


def create_mock_treasury_state(
    as_of_iso: Optional[str] = None,
    scenario: str = "NORMAL",
) -> TreasuryState:
    """Generates a realistic multi-asset TreasuryState."""
    now = as_of_iso or datetime.now(timezone.utc).isoformat()

    # Base assets
    assets = [
        AssetPosition(
            asset_id="CASH-USD-FED",
            name="Federal Reserve Bank Operating Cash",
            asset_class="CASH",
            market_value=45_000_000.0,
            accessible_liquidity=45_000_000.0,
            settlement_days=0,
            yield_pct=5.25,
            haircut_pct=0.0,
            is_eligible=True,
            allocation_cap=100_000_000.0,
        ),
        AssetPosition(
            asset_id="REPO-OVN-SOV",
            name="Overnight Tri-Party Sovereign Repo",
            asset_class="REPO",
            market_value=30_000_000.0,
            accessible_liquidity=29_850_000.0,
            settlement_days=0,
            yield_pct=5.35,
            duration_years=0.01,
            haircut_pct=0.5,
            is_eligible=True,
            allocation_cap=50_000_000.0,
        ),
        AssetPosition(
            asset_id="REPO-30D-GCF",
            name="30-Day General Collateral Finance Repo",
            asset_class="REPO",
            market_value=25_000_000.0,
            accessible_liquidity=24_500_000.0,
            settlement_days=0,
            yield_pct=5.50,
            duration_years=0.08,
            haircut_pct=2.0,
            is_eligible=True,
            allocation_cap=40_000_000.0,
        ),
        AssetPosition(
            asset_id="SOV-UST-2Y",
            name="US Treasury Note 2-Year Benchmark",
            asset_class="SOVEREIGN_BONDS",
            market_value=60_000_000.0,
            accessible_liquidity=57_000_000.0,
            settlement_days=1,
            yield_pct=4.75,
            duration_years=1.92,
            haircut_pct=5.0,
            is_eligible=True,
            allocation_cap=80_000_000.0,
        ),
        AssetPosition(
            asset_id="SOV-UST-10Y",
            name="US Treasury Note 10-Year Benchmark",
            asset_class="SOVEREIGN_BONDS",
            market_value=40_000_000.0,
            accessible_liquidity=36_800_000.0,
            settlement_days=1,
            yield_pct=4.40,
            duration_years=8.25,
            haircut_pct=8.0,
            is_eligible=True,
            allocation_cap=60_000_000.0,
        ),
        AssetPosition(
            asset_id="EQ-DIV-CORE",
            name="S&P 500 Dividend Aristocrats ETF",
            asset_class="EQUITIES",
            market_value=20_000_000.0,
            accessible_liquidity=17_000_000.0,
            settlement_days=2,
            yield_pct=3.10,
            haircut_pct=15.0,
            is_eligible=True,
            allocation_cap=25_000_000.0,
        ),
    ]

    obligations = [
        Obligation(
            obligation_id="OBL-01",
            due_timestamp=(datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
            amount=12_000_000.0,
            description="CHIPS / Fedwire Settlement Clearing Batch",
            priority="MANDATORY",
            beneficiary="Clearing House Interbank Payments",
        ),
        Obligation(
            obligation_id="OBL-02",
            due_timestamp=(datetime.now(timezone.utc) + timedelta(hours=18)).isoformat(),
            amount=8_500_000.0,
            description="Institutional Client Structured Term Withdrawal",
            priority="MANDATORY",
            beneficiary="Asset Management Corp",
        ),
    ]

    return TreasuryState(
        timestamp=now,
        available_cash=45_000_000.0,
        asset_positions=assets,
        obligations=obligations,
        authorized_credit_availability=50_000_000.0,
        data_quality_warnings=[],
        protected_reserve_floor=25_000_000.0,
        core_capital_pct=0.75,
        hot_capital_pct=0.25,
    )


def create_mock_forecast_bundle(
    as_of_iso: Optional[str] = None,
    scenario: str = "NORMAL",
) -> ForecastBundle:
    """Generates realistic multi-horizon forecasts with uncertainty intervals."""
    now = as_of_iso or datetime.now(timezone.utc).isoformat()

    multiplier = 1.0
    if scenario == "BANKING_STRESS":
        multiplier = 2.4
    elif scenario == "RATE_HIKE_SHOCK":
        multiplier = 1.6

    forecasts = {
        "1h": HorizonForecast(
            horizon="1h",
            expected_inflow=3_000_000.0,
            expected_outflow=5_000_000.0 * multiplier,
            expected_net_flow=(3_000_000.0 - 5_000_000.0 * multiplier),
            uncertainty_p05=-(7_500_000.0 * multiplier),
            uncertainty_p50=(3_000_000.0 - 5_000_000.0 * multiplier),
            uncertainty_p95=1_000_000.0,
            confidence_score=0.92,
        ),
        "6h": HorizonForecast(
            horizon="6h",
            expected_inflow=8_000_000.0,
            expected_outflow=16_000_000.0 * multiplier,
            expected_net_flow=(8_000_000.0 - 16_000_000.0 * multiplier),
            uncertainty_p05=-(22_000_000.0 * multiplier),
            uncertainty_p50=(8_000_000.0 - 16_000_000.0 * multiplier),
            uncertainty_p95=-2_000_000.0,
            confidence_score=0.88,
        ),
        "24h": HorizonForecast(
            horizon="24h",
            expected_inflow=18_000_000.0,
            expected_outflow=32_000_000.0 * multiplier,
            expected_net_flow=(18_000_000.0 - 32_000_000.0 * multiplier),
            uncertainty_p05=-(42_000_000.0 * multiplier),
            uncertainty_p50=(18_000_000.0 - 32_000_000.0 * multiplier),
            uncertainty_p95=-5_000_000.0,
            confidence_score=0.84,
        ),
        "48h": HorizonForecast(
            horizon="48h",
            expected_inflow=35_000_000.0,
            expected_outflow=50_000_000.0 * multiplier,
            expected_net_flow=(35_000_000.0 - 50_000_000.0 * multiplier),
            uncertainty_p05=-(65_000_000.0 * multiplier),
            uncertainty_p50=(35_000_000.0 - 50_000_000.0 * multiplier),
            uncertainty_p95=-10_000_000.0,
            confidence_score=0.78,
        ),
        "72h": HorizonForecast(
            horizon="72h",
            expected_inflow=52_000_000.0,
            expected_outflow=70_000_000.0 * multiplier,
            expected_net_flow=(52_000_000.0 - 70_000_000.0 * multiplier),
            uncertainty_p05=-(88_000_000.0 * multiplier),
            uncertainty_p50=(52_000_000.0 - 70_000_000.0 * multiplier),
            uncertainty_p95=-12_000_000.0,
            confidence_score=0.72,
        ),
    }

    return ForecastBundle(
        timestamp=now,
        point_forecasts=forecasts,
        scenarios={
            "BASELINE": {"24h": -14_000_000.0},
            "DEPOSIT_RUN": {"24h": -38_000_000.0},
            "RATE_HIKE_STRESS": {"24h": -22_000_000.0},
        },
        quality_status="HIGH",
        model_versions={"deposit_model": "v2.1", "ensemble_combiner": "v1.4"},
    )


def create_mock_decision_record(
    as_of_iso: Optional[str] = None,
    scenario: str = "NORMAL",
) -> DecisionRecord:
    """Generates a realistic Next Best Action DecisionRecord."""
    now = as_of_iso or datetime.now(timezone.utc).isoformat()
    cid = f"DEC-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    gates = [
        HardGateOutcome(
            gate_name="MandatoryReserveFloorGate",
            passed=True,
            threshold_value=25_000_000.0,
            observed_value=35_000_000.0,
            reason="Post-action cash and intraday liquidity exceed mandatory statutory floor.",
        ),
        HardGateOutcome(
            gate_name="CreditAvailabilityGate",
            passed=True,
            threshold_value=50_000_000.0,
            observed_value=50_000_000.0,
            reason="Authorized credit facility fully intact.",
        ),
        HardGateOutcome(
            gate_name="AssetEligibilityGate",
            passed=True,
            threshold_value=1.0,
            observed_value=1.0,
            reason="Selected asset qualifies under Basel III High-Quality Liquid Assets (HQLA) Level 1.",
        ),
    ]

    alternatives = [
        AlternativeAction(
            action_type="DRAW_CREDIT_FACILITY",
            target_asset="COMMITTED_REVOLVER",
            amount=15_000_000.0,
            estimated_total_cost=22_500.0,
            settlement_horizon_hours=2,
            rejection_reason="Commitment fee and draw spread exceed repo financing cost by 45 bps.",
            net_yield_impact=-0.0045,
        ),
        AlternativeAction(
            action_type="LIQUIDATE_SOVEREIGN_BOND",
            target_asset="SOV-UST-2Y",
            amount=15_000_000.0,
            estimated_total_cost=18_200.0,
            settlement_horizon_hours=24,
            rejection_reason="T+1 settlement latency creates timing gap for T+0 intraday clearing obligations.",
            net_yield_impact=-0.0036,
        ),
        AlternativeAction(
            action_type="HOLD_CASH_AND_DEFICIT",
            target_asset="CASH-USD-FED",
            amount=0.0,
            estimated_total_cost=75_000.0,
            settlement_horizon_hours=0,
            rejection_reason="Incurs daylight overdraft penalty fees and impairs Treasury Safety Score.",
            net_yield_impact=-0.0150,
        ),
    ]

    return DecisionRecord(
        decision_id=cid,
        timestamp=now,
        selected_action="DRAW_REPO_30D",
        target_asset="REPO-30D-GCF",
        amount=15_000_000.0,
        reserve=25_000_000.0,
        eligible_surplus=20_000_000.0,
        estimated_costs={
            "interest": 6_875.0,
            "tax": 1_250.0,
            "liquidation_penalty": 0.0,
            "slippage": 450.0,
            "total": 8_575.0,
        },
        hard_gate_outcomes=gates,
        model_versions={"decision_engine": "v1.2.0", "optimizer": "v2.0.1"},
        reason_codes=["MIN_COST_LIQUIDITY", "RESERVE_PRESERVATION", "T0_SETTLEMENT_ALIGNED"],
        alternatives=alternatives,
        treasury_safety_score=86.5,
        tss_factors={
            "liquidity_buffer": 38.0,
            "volatility_penalty": -4.5,
            "macro_event_penalty": -7.0,
            "credit_cushion": 20.0,
        },
    )
