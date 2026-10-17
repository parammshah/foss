"""
Stress Lab Dashboard View: Interactive Scenario Stress Testing & Liquidity Firewall Simulation.
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from reserve_juggler.integration.facade import ApplicationFacade, DashboardSnapshot


def render_stress_lab(snapshot: DashboardSnapshot, facade: ApplicationFacade) -> None:
    st.markdown("## 🔬 Liquidity Stress Lab")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.9rem;'>Interactive stress testing engine: simulate withdrawal spikes, bond yield jumps, repo rollover delays, and credit freezing against the Future-Protected Capital Firewall.</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
        st.markdown("### 🎛️ Configure Macro Shocks")
        deposit_shock = st.slider("Withdrawal Surge (% over baseline)", min_value=0, max_value=250, value=50, step=10)
        yield_shock = st.slider("Sovereign Bond Yield Shock (+ bps)", min_value=0, max_value=200, value=40, step=10)
        repo_delay = st.slider("Repo Settlement Latency (Days Delay)", min_value=0, max_value=5, value=1)
        credit_freeze = st.checkbox("Simulate Bridge Credit Facility Unavailability (Credit Freeze)", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    # Run stress test simulation via facade
    res = facade.run_stress_lab_scenario(
        deposit_shock_pct=float(deposit_shock),
        rate_shock_bps=float(yield_shock),
        repo_delay_days=repo_delay,
        credit_unavailable=credit_freeze,
    )

    with c2:
        st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Stress Test Impact Results")
        
        sf_badge = "badge-critical" if res["is_shortfall"] else "badge-ok"
        status_text = f"CRITICAL DEFICIT (${res['shortfall_amount']:,.2f})" if res["is_shortfall"] else "SURPLUS PROTECTED"
        
        st.markdown(
            f"""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                <span style='font-size: 1.1rem; font-weight: 700;'>Reserve Floor Status:</span>
                <span class='{sf_badge}'>{status_text}</span>
            </div>
            <div style='background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 12px; margin-bottom: 12px;'>
                <div>Projected Stress Outflow: <b>${res['simulated_outflow']:,.2f}</b></div>
                <div>Bond Portfolio MTM Haircut/Loss: <b>${res['bond_mtm_loss']:,.2f}</b></div>
                <div>Post-Shock Operating Cash: <b>${res['post_shock_cash']:,.2f}</b></div>
                <div>Statutory Mandatory Reserve Floor: <b>${res['reserve_floor']:,.2f}</b></div>
            </div>
            <div class='rj-callout'>
                <b>Recommended Defensive Liquidity Hedge:</b> <code>{res['recommended_hedge']}</code><br/>
                Activates emergency overnight liquidity backstop to prevent core reserve breach.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Waterfall Chart comparing Baseline Cash vs Shocks vs Post-Shock
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 📉 Liquidity Waterfall Under Shock")
    
    cash_start = snapshot.treasury_state.available_cash / 1e6
    outflow_m = res["simulated_outflow"] / 1e6
    bond_loss_m = res["bond_mtm_loss"] / 1e6
    cash_end = res["post_shock_cash"] / 1e6
    reserve_m = res["reserve_floor"] / 1e6

    fig = go.Figure(go.Waterfall(
        name="Liquidity",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Starting Cash", "Withdrawal Outflows", "Bond MTM Shock", "Remaining Cash"],
        y=[cash_start, -outflow_m, -bond_loss_m, cash_end],
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
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
