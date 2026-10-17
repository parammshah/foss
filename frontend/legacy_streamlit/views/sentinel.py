"""
Macro-Panic Sentinel Dashboard View: Real-Time Financial Intelligence Monitoring.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
from reserve_juggler.integration.facade import ApplicationFacade, DashboardSnapshot


def render_sentinel(snapshot: DashboardSnapshot, facade: ApplicationFacade) -> None:
    st.markdown("## 📡 Macro-Panic Sentinel Intelligence")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.9rem;'>External financial-event monitoring across central bank releases, market shocks, bank funding stress, and corroborated withdrawal velocity.</p>",
        unsafe_allow_html=True,
    )

    sent = snapshot.sentinel_result

    # Mode and Health Bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class='rj-metric-tile'>
                <div class='rj-metric-label'>Feed Ingestion Mode</div>
                <div class='rj-metric-val' style='font-size: 1.25rem;'>{sent.mode}</div>
                <div class='rj-metric-sub'>Fault-Tolerant Feed Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class='rj-metric-tile'>
                <div class='rj-metric-label'>Events Ingested</div>
                <div class='rj-metric-val' style='font-size: 1.25rem;'>{sent.total_events_processed}</div>
                <div class='rj-metric-sub'>De-duplicated: {sent.duplicates_filtered}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        stress_color = "#EF4444" if sent.aggregate_macro_stress_score >= 60 else ("#F59E0B" if sent.aggregate_macro_stress_score >= 35 else "#10B981")
        st.markdown(
            f"""
            <div class='rj-metric-tile'>
                <div class='rj-metric-label'>Macro Stress Index</div>
                <div class='rj-metric-val' style='font-size: 1.25rem; color: {stress_color};'>{sent.aggregate_macro_stress_score:.1f}</div>
                <div class='rj-metric-sub'>Scale: 0 (Calm) - 100 (Crisis)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class='rj-metric-tile'>
                <div class='rj-metric-label'>Recommended Scenarios</div>
                <div class='rj-metric-val' style='font-size: 1.1rem; color: #818CF8;'>{len(sent.recommended_stress_scenarios)}</div>
                <div class='rj-metric-sub'>{", ".join(sent.recommended_stress_scenarios[:2]) if sent.recommended_stress_scenarios else 'None'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Active Intelligence Signals Table
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 🔔 Active Risk Signals & Treasury Exposure Relevance")
    
    if sent.active_signals:
        rows = []
        for s in sent.active_signals:
            rows.append({
                "Signal ID": s.signal_id,
                "Source": s.source,
                "Category": s.category,
                "Severity": s.severity,
                "Confidence": f"{s.confidence:.0%}",
                "Corroboration": s.corroboration_status,
                "Sentiment Score": f"{s.sentiment_score:+.2f}",
                "Exposure Relevance": f"{s.relevance_to_treasury:.0%}",
                "Affected Markets": ", ".join(s.affected_markets),
                "Evidence Summary": s.evidence[:110] + "...",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No active risk signals triggered.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Inject Custom Event Test
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 🧪 Inject Synthetic Financial Event (Lab Demonstration)")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.85rem;'>Test real-time event interpretation, deduplication clustering, and safety guardrails against sudden news flashes.</p>",
        unsafe_allow_html=True,
    )

    with st.form("inject_event_form"):
        ea, eb = st.columns([2, 1])
        with ea:
            ev_title = st.text_input("Event Headline", value="Central Bank Announces Unscheduled Emergency Liquidity Facility")
            ev_summary = st.text_area("Event Summary", value="Regulatory authorities open discount window with waived collateral haircuts in response to interbank settlement tightness.")
        with eb:
            ev_cat = st.selectbox("Category", ["BANKING_STRESS", "MONETARY_POLICY", "MARKET_SHOCK", "WITHDRAWAL_ACCELERATION"])
            ev_source = st.selectbox("Source", ["CENTRAL_BANK_BULLETIN", "FINANCIAL_TIMES_BREAKING", "SOCIAL_MEDIA_RUMOR", "INTERBANK_BROKER_DESK"])
            ev_cred = 0.98 if "CENTRAL_BANK" in ev_source else (0.88 if "FINANCIAL_TIMES" in ev_source else 0.35)

        inject_btn = st.form_submit_button("Inject Event to Sentinel")
        if inject_btn:
            from reserve_juggler.events.models import RawEventRecord
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            
            custom_ev = RawEventRecord(
                event_id=f"INJ-{int(datetime.now().timestamp())}",
                source=ev_source,
                title=ev_title,
                summary=ev_summary,
                publication_time=now,
                ingestion_time=now,
                category=ev_cat,
                affected_markets=["CASH", "REPO", "RATES"],
                evidence_url="https://emergency.centralbank.example/notice",
                source_credibility=ev_cred,
                sentiment_score=-0.75,
            )
            facade.sentinel.inject_synthetic_event(custom_ev)
            st.success("Custom event successfully injected into Sentinel processing pipeline!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
