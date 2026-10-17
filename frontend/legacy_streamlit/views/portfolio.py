"""
Portfolio Dashboard View: Asset Allocations, Marked-to-Market vs Accessible Liquidity.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from reserve_juggler.integration.facade import DashboardSnapshot


def render_portfolio(snapshot: DashboardSnapshot) -> None:
    st.markdown("## 💼 Multi-Asset Treasury Portfolio")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.9rem;'>Cross-asset holdings across cash, repo, sovereign bonds, and equities with marked-to-market valuations, haircut discounts, and settlement timelines.</p>",
        unsafe_allow_html=True,
    )

    state = snapshot.treasury_state

    # Comparison Bar Chart: Market Value vs Accessible Liquidity
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Marked-to-Market vs Accessible Liquidity")

    names = ["Operating Cash"] + [p.name.split()[0] + " " + p.asset_class for p in state.asset_positions]
    mtm_vals = [state.available_cash / 1e6] + [p.market_value / 1e6 for p in state.asset_positions]
    accessible_vals = [state.available_cash / 1e6] + [p.accessible_liquidity / 1e6 for p in state.asset_positions]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=mtm_vals,
        name="Marked-to-Market ($M)",
        marker_color="#6366F1",
    ))
    fig.add_trace(go.Bar(
        x=names,
        y=accessible_vals,
        name="Accessible Intraday Liquidity ($M)",
        marker_color="#10B981",
    ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#CBD5E1"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="$ Millions", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=20, b=30, l=40, r=20),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Position Table
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 📑 Asset Positions & Settlement Constraints")

    rows = []
    # Operating cash row
    rows.append({
        "Asset Name": "Federal Reserve Operating Cash",
        "Class": "CASH",
        "MTM Value": f"${state.available_cash:,.2f}",
        "Accessible Liquidity": f"${state.available_cash:,.2f}",
        "Haircut": "0.0%",
        "Yield": "5.25%",
        "Settlement": "T+0 (Immediate)",
        "Duration": "0.0 yrs",
        "Eligible": "✅ Yes",
    })

    for p in state.asset_positions:
        rows.append({
            "Asset Name": p.name,
            "Class": p.asset_class,
            "MTM Value": f"${p.market_value:,.2f}",
            "Accessible Liquidity": f"${p.accessible_liquidity:,.2f}",
            "Haircut": f"{p.haircut_pct:.1f}%",
            "Yield": f"{p.yield_pct:.2f}%",
            "Settlement": f"T+{p.settlement_days}",
            "Duration": f"{p.duration_years:.2f} yrs",
            "Eligible": "✅ Yes" if p.is_eligible else "❌ Restricted",
        })

    df_pos = pd.DataFrame(rows)
    st.dataframe(df_pos, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
