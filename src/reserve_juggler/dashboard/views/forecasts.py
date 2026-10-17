"""
Forecasts and Reliability Dashboard View: Multi-Horizon Predictions and Error Tracking.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from reserve_juggler.integration.facade import DashboardSnapshot


def render_forecasts(snapshot: DashboardSnapshot) -> None:
    st.markdown("## 📈 Cash Flow Forecasts & Model Calibration")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.9rem;'>Multi-horizon ML predictions with OR-based combination, confidence intervals, and calibration drift diagnostics.</p>",
        unsafe_allow_html=True,
    )

    bundle = snapshot.forecast_bundle
    perf = snapshot.system_performance.forecast_metrics

    # Model Status and Quality Bar
    q_badge = "badge-ok" if bundle.quality_status == "HIGH" else "badge-high"
    d_badge = "badge-critical" if perf.drift_detected else "badge-ok"
    
    st.markdown(
        f"""
        <div style='display: flex; gap: 15px; margin-bottom: 20px;'>
            <div class='rj-card' style='flex: 1; margin-bottom: 0;'>
                <div class='rj-metric-label'>Forecast Data Quality</div>
                <div style='margin-top: 6px;'><span class='{q_badge}'>{bundle.quality_status}</span></div>
            </div>
            <div class='rj-card' style='flex: 1; margin-bottom: 0;'>
                <div class='rj-metric-label'>System Drift Status</div>
                <div style='margin-top: 6px;'><span class='{d_badge}'>{'DRIFT DETECTED' if perf.drift_detected else 'STABLE NO DRIFT'}</span></div>
            </div>
            <div class='rj-card' style='flex: 1; margin-bottom: 0;'>
                <div class='rj-metric-label'>Mean Absolute % Error (MAPE)</div>
                <div class='rj-metric-val' style='font-size: 1.3rem;'>{perf.mape:.1f}%</div>
            </div>
            <div class='rj-card' style='flex: 1; margin-bottom: 0;'>
                <div class='rj-metric-label'>90% Interval Coverage</div>
                <div class='rj-metric-val' style='font-size: 1.3rem; color: #10B981;'>{perf.interval_coverage_pct:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Multi-Horizon Forecast Chart with Uncertainty Bands
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 🌊 Net Cash Flow Projections Across Horizons")

    horizons = ["1h", "6h", "24h", "48h", "72h"]
    expected_flows = [bundle.point_forecasts[h].expected_net_flow / 1e6 for h in horizons if h in bundle.point_forecasts]
    p05_flows = [bundle.point_forecasts[h].uncertainty_p05 / 1e6 for h in horizons if h in bundle.point_forecasts]
    p95_flows = [bundle.point_forecasts[h].uncertainty_p95 / 1e6 for h in horizons if h in bundle.point_forecasts]

    fig = go.Figure()

    # Upper bound (p95)
    fig.add_trace(go.Scatter(
        x=horizons,
        y=p95_flows,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="p95 Optimistic",
    ))

    # Lower bound (p05) with fill
    fig.add_trace(go.Scatter(
        x=horizons,
        y=p05_flows,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(99, 102, 241, 0.18)",
        name="90% Uncertainty Interval [p05, p95]",
    ))

    # Expected trajectory
    fig.add_trace(go.Scatter(
        x=horizons,
        y=expected_flows,
        mode="lines+markers",
        line=dict(color="#6366F1", width=3),
        marker=dict(size=8, color="#818CF8"),
        name="Expected Net Flow (OR Combined)",
    ))

    # Zero Line
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#CBD5E1"),
        xaxis=dict(title="Liquidity Horizon", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Net Cash Flow ($ Millions)", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=20, b=30, l=40, r=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Breakdown Table by Horizon
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Horizon Inflows, Outflows & Confidence")
    
    rows = []
    for h in horizons:
        if h in bundle.point_forecasts:
            fc = bundle.point_forecasts[h]
            rows.append({
                "Horizon": fc.horizon,
                "Expected Inflows": f"${fc.expected_inflow:,.2f}",
                "Expected Outflows": f"${fc.expected_outflow:,.2f}",
                "Net Flow": f"${fc.expected_net_flow:,.2f}",
                "p05 Stress Outflow": f"${abs(fc.uncertainty_p05):,.2f}",
                "Confidence Score": f"{fc.confidence_score:.1%}",
            })
    df_fc = pd.DataFrame(rows)
    st.dataframe(df_fc, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
