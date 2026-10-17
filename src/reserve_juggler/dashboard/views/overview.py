"""
Overview Dashboard View: Executive Treasury Intelligence Summary.
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from reserve_juggler.integration.facade import DashboardSnapshot


def render_overview(snapshot: DashboardSnapshot) -> None:
    st.markdown("## 🏛️ Treasury Executive Overview")
    st.markdown(
        f"<p style='color: #94A3B8; font-size: 0.9rem;'>As of: <code>{snapshot.as_of_timestamp}</code> | Operational Mode: <b>{snapshot.sentinel_result.mode}</b> | Scenario: <span class='badge-neutral'>{snapshot.active_scenario}</span></p>",
        unsafe_allow_html=True,
    )

    state = snapshot.treasury_state
    total_assets = sum(p.market_value for p in state.asset_positions)
    total_portfolio = total_assets + state.available_cash
    reserve = state.protected_reserve_floor
    eligible_surplus = max(0.0, total_portfolio - reserve)

    # 4 Primary Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class='rj-metric-tile'>
                <div class='rj-metric-label'>Available Operational Cash</div>
                <div class='rj-metric-val'>${state.available_cash / 1e6:.1f}M</div>
                <div class='rj-metric-sub'>Intraday Fedwire Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class='rj-metric-tile'>
                <div class='rj-metric-label'>Protected Reserve Floor</div>
                <div class='rj-metric-val' style='color: #F59E0B;'>${reserve / 1e6:.1f}M</div>
                <div class='rj-metric-sub'>Mandatory Capital Firewall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class='rj-metric-tile'>
                <div class='rj-metric-label'>Eligible Surplus Capital</div>
                <div class='rj-metric-val' style='color: #10B981;'>${eligible_surplus / 1e6:.1f}M</div>
                <div class='rj-metric-sub'>Yield Optimization Pool</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        tss = snapshot.latest_decision.treasury_safety_score
        tss_color = "#10B981" if tss >= 80 else ("#F59E0B" if tss >= 65 else "#EF4444")
        st.markdown(
            f"""
            <div class='rj-metric-tile'>
                <div class='rj-metric-label'>Treasury Safety Score</div>
                <div class='rj-metric-val' style='color: {tss_color};'>{tss:.1f}/100</div>
                <div class='rj-metric-sub'>Multi-Factor Risk Health</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Secondary Row: Next Best Action & Macro-Panic Sentinel Alert Status
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Recommended Next Best Action")
        dec = snapshot.latest_decision
        exp = snapshot.decision_explanation

        action_display = dec.selected_action.replace("_", " ").title()
        st.markdown(
            f"""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <span style='font-size: 1.35rem; font-weight: 700; color: #6366F1;'>{action_display}</span>
                <span class='badge-neutral'>Target: {dec.target_asset}</span>
            </div>
            <div style='font-size: 1.1rem; font-family: monospace; color: #E2E8F0; margin-bottom: 8px;'>
                Recommended Amount: <b>${dec.amount:,.2f}</b>
            </div>
            <div style='font-size: 0.88rem; color: #94A3B8; margin-bottom: 12px;'>
                Estimated Cost: <b>${dec.estimated_costs.get('total', 0.0):,.2f}</b> | 
                Settlement Horizon: <b>{exp.settlement_days} Day(s) ({exp.settlement_note})</b>
            </div>
            <div class='rj-callout'>
                <b>Decision Rationale:</b> {exp.primary_justification}<br/>
                <span style='font-size: 0.82rem; color: #CBD5E1;'>{exp.grounded_narrative.splitlines()[0] if exp.grounded_narrative else ''}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
        st.markdown("### 🚨 Macro-Panic Sentinel Status")
        sent = snapshot.sentinel_result
        stress_score = sent.aggregate_macro_stress_score
        stress_badge = "badge-critical" if stress_score >= 60 else ("badge-high" if stress_score >= 35 else "badge-ok")

        st.markdown(
            f"""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <span style='color: #94A3B8; font-size: 0.88rem;'>Aggregate Macro Stress Index</span>
                <span class='{stress_badge}'>{stress_score:.1f} / 100</span>
            </div>
            <div style='font-size: 0.85rem; color: #CBD5E1; margin-bottom: 10px;'>
                Active Signals: <b>{len(sent.active_signals)}</b> | Duplicates Filtered: <b>{sent.duplicates_filtered}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if sent.active_signals:
            for s in sent.active_signals[:2]:
                sev_badge = "badge-critical" if s.severity == "CRITICAL" else ("badge-high" if s.severity == "HIGH" else "badge-medium")
                st.markdown(
                    f"""
                    <div style='padding: 8px 12px; background: rgba(30, 41, 59, 0.5); border-radius: 8px; margin-bottom: 6px; border-left: 3px solid #6366F1;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <span style='font-weight: 600; font-size: 0.82rem; color: #F1F5F9;'>{s.source}</span>
                            <span class='{sev_badge}'>{s.severity}</span>
                        </div>
                        <div style='font-size: 0.78rem; color: #94A3B8; margin-top: 3px;'>{s.evidence[:95]}...</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Market conditions stable. No elevated macroeconomic risk alerts.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Capital Allocation Breakdown Chart
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Capital Structure & Reserve Partitioning")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        # Donut Chart for Asset Class Distribution
        labels = ["Cash"] + [p.name for p in state.asset_positions]
        values = [state.available_cash] + [p.market_value for p in state.asset_positions]
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=["#3B82F6", "#6366F1", "#8B5CF6", "#EC4899", "#F59E0B", "#10B981"]),
            textinfo="percent",
            hoverinfo="label+value+percent",
        )])
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1", size=11),
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        st.markdown("#### Core vs Hot Capital Firewall")
        st.markdown(
            f"""
            - **Core Capital ({state.core_capital_pct:.0%}):** ${total_portfolio * state.core_capital_pct / 1e6:.1f}M dedicated to structural reserves, high-quality collateral, and regulatory baseline.
            - **Hot Capital ({state.hot_capital_pct:.0%}):** ${total_portfolio * state.hot_capital_pct / 1e6:.1f}M flexible tactical capital allocated to intraday repo optimization and term yield capture.
            - **Protected Reserve Floor:** ${reserve / 1e6:.1f}M hard statutory firewall (non-negotiable invariant).
            - **Authorized Credit Facility:** ${state.authorized_credit_availability / 1e6:.1f}M standby buffer.
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)
