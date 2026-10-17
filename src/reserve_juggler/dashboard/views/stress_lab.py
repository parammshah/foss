"""
Stress Lab Dashboard View: Interactive Scenario Stress Testing & 3-Strategy Comparative Benchmark.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from reserve_juggler.integration.facade import ApplicationFacade, DashboardSnapshot


def render_stress_lab(snapshot: DashboardSnapshot, facade: ApplicationFacade) -> None:
    st.markdown("## 🔬 Liquidity Stress Lab")
    st.markdown(
        f"<p style='color: #94A3B8; font-size: 0.9rem;'>Simulates multi-asset treasury dynamics and compares 3 benchmark strategies under identical macro shocks against the Capital Firewall. "
        f"Data Mode: <b>{snapshot.data_source_mode}</b></p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
        st.markdown("### 🎛️ Configure Macro Shocks")
        deposit_shock = st.slider("Withdrawal Surge (% over baseline)", min_value=0, max_value=250, value=50, step=10)
        yield_shock = st.slider("Sovereign Bond Yield Shock (+ bps)", min_value=0, max_value=200, value=40, step=10)
        equity_drop = st.slider("Equity Market Drawdown (- %)", min_value=0, max_value=40, value=15, step=5)
        repo_delay = st.slider("Repo Settlement Latency (Days Delay)", min_value=0, max_value=5, value=1)
        credit_freeze = st.checkbox("Simulate Bridge Credit Facility Unavailability (Credit Freeze)", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    # Run stress test simulation via facade (which routes to StressLabSimulationEngine)
    res = facade.run_stress_lab_scenario(
        deposit_shock_pct=float(deposit_shock),
        rate_shock_bps=float(yield_shock),
        repo_delay_days=repo_delay,
        credit_unavailable=credit_freeze,
        equity_drawdown_pct=float(equity_drop),
    )

    rj_res = res["strategies"]["reserve_juggler"]
    sf_badge = "badge-critical" if rj_res.reserve_breach else "badge-ok"
    status_text = f"CRITICAL DEFICIT (${rj_res.liquidity_shortfall:,.2f})" if rj_res.reserve_breach else "SURPLUS PROTECTED"

    with c2:
        st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Reserve Juggler Impact Results")
        st.markdown(
            f"""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                <span style='font-size: 1.1rem; font-weight: 700;'>Reserve Floor Status:</span>
                <span class='{sf_badge}'>{status_text}</span>
            </div>
            <div style='background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 12px; margin-bottom: 12px;'>
                <div>Projected Stress Outflow: <b>${rj_res.total_outflows:,.2f}</b></div>
                <div>Bond Portfolio MTM Loss: <b>${rj_res.bond_mtm_loss:,.2f}</b></div>
                <div>Equity Portfolio Loss: <b>${rj_res.equity_loss:,.2f}</b></div>
                <div>Post-Shock Operating Cash: <b>${rj_res.final_cash:,.2f}</b></div>
                <div>Statutory Mandatory Reserve Floor: <b>${res['parameters']['reserve_floor']:,.2f}</b></div>
            </div>
            <div class='rj-callout'>
                <b>Recommended Defensive Liquidity Hedge:</b> <code>{res['recommended_hedge']}</code><br/>
                Activates overnight repo liquidity backstop to prevent core reserve breach.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Strategy Comparison Table (Fair, identical starting conditions)
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Fair Benchmark Strategy Comparison")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.85rem;'>Evaluated under identical initial portfolio assets, obligations, and macro stress parameters.</p>",
        unsafe_allow_html=True,
    )

    strat_rows = []
    for key in ("all_cash", "greedy_yield", "reserve_juggler"):
        s = res["strategies"][key]
        strat_rows.append({
            "Strategy": s.strategy_name,
            "Starting Cash": f"${s.starting_cash / 1e6:.1f}M",
            "Final Cash": f"${s.final_cash / 1e6:.1f}M",
            "Bond/Eq Loss": f"-${(s.bond_mtm_loss + s.equity_loss) / 1e6:.2f}M",
            "Emergency Draw": f"${s.emergency_credit_drawn / 1e6:.1f}M",
            "Reserve Breach": "❌ BREACH" if s.reserve_breach else "✅ PROTECTED",
            "Shortfall Deficit": f"${s.liquidity_shortfall / 1e6:.2f}M" if s.liquidity_shortfall > 0 else "$0.00",
            "Strategy Rationale": s.strategy_description,
        })

    df_strats = pd.DataFrame(strat_rows)
    st.dataframe(df_strats, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Waterfall Chart for Reserve Juggler
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 📉 Reserve Juggler Liquidity Waterfall")
    
    cash_start = rj_res.starting_cash / 1e6
    inflow_m = rj_res.total_inflows / 1e6
    outflow_m = rj_res.total_outflows / 1e6
    cash_end = rj_res.final_cash / 1e6
    reserve_m = res["parameters"]["reserve_floor"] / 1e6

    fig = go.Figure(go.Waterfall(
        name="Liquidity",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Starting Cash", "Operational Inflows", "Stress Outflows", "Remaining Cash"],
        y=[cash_start, inflow_m, -outflow_m, cash_end],
        connector={"line": {"color": "#94A3B8"}},
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#10B981"}},
        totals={"marker": {"color": "#6366F1"}},
    ))

    fig.add_hline(y=reserve_m, line_dash="dash", line_color="#F59E0B", annotation_text="Mandatory Reserve Floor ($25M)")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#CBD5E1"),
        yaxis=dict(title="$ Millions", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=20, b=30, l=40, r=20),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
