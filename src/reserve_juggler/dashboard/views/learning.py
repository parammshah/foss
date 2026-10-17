"""
System Learning Dashboard View: Lifecycle Performance, Candidate Updates & Audited Rollback.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
from reserve_juggler.integration.contracts import UserRole
from reserve_juggler.integration.facade import ApplicationFacade, DashboardSnapshot


def render_learning(snapshot: DashboardSnapshot, facade: ApplicationFacade) -> None:
    st.markdown("## 🧠 System-Wide Self-Learning Feedback Loop")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.9rem;'>Cross-subsystem telemetry observing outcomes from ingestion to human approval, with safety invariants, shadow-mode evaluation, and authorized version rollback.</p>",
        unsafe_allow_html=True,
    )

    perf = snapshot.system_performance
    learning_service = facade.learning_service

    # Top Health Banner
    health_badge = "badge-ok" if perf.overall_system_health == "HEALTHY" else (
        "badge-critical" if perf.overall_system_health == "ACTION_REQUIRED" else "badge-high"
    )
    st.markdown(
        f"""
        <div style='display: flex; gap: 15px; margin-bottom: 20px;'>
            <div class='rj-card' style='flex: 1; margin-bottom: 0;'>
                <div class='rj-metric-label'>Overall System Health</div>
                <div style='margin-top: 6px;'><span class='{health_badge}'>{perf.overall_system_health}</span></div>
            </div>
            <div class='rj-card' style='flex: 1; margin-bottom: 0;'>
                <div class='rj-metric-label'>Total Recorded Events</div>
                <div class='rj-metric-val' style='font-size: 1.3rem;'>{perf.total_learning_events}</div>
            </div>
            <div class='rj-card' style='flex: 1; margin-bottom: 0;'>
                <div class='rj-metric-label'>Liquidity Shortfalls Count</div>
                <div class='rj-metric-val' style='font-size: 1.3rem; color: #10B981;'>{perf.decision_metrics.total_shortfalls}</div>
            </div>
            <div class='rj-card' style='flex: 1; margin-bottom: 0;'>
                <div class='rj-metric-label'>Cost Slippage Avg</div>
                <div class='rj-metric-val' style='font-size: 1.3rem; color: #38BDF8;'>{perf.decision_metrics.avg_cost_slippage_bps:.1f} bps</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if perf.inconclusive_reason:
        st.info(f"ℹ️ {perf.inconclusive_reason}")

    # Active Subsystem Versions Stack
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 📦 Active Production Model & Policy Versions")
    v_cols = st.columns(len(perf.active_versions))
    for col, (comp, ver) in zip(v_cols, perf.active_versions.items()):
        with col:
            st.markdown(
                f"""
                <div style='background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 8px; text-align: center;'>
                    <div style='font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;'>{comp}</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: #6366F1;'>{ver}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # Candidate Updates & Shadow Mode
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 🧪 Candidate Updates & Shadow-Mode Validation")
    candidates = snapshot.active_candidates

    if candidates:
        cand_rows = []
        for c in candidates:
            cand_rows.append({
                "Candidate ID": c.candidate_id,
                "Component": c.component,
                "Parameter": c.parameter_key,
                "Proposed Value": str(c.proposed_value),
                "Status": c.status.value,
                "Holdout Score": f"{c.holdout_evaluation_score:.2f}",
                "Stress Shortfalls": c.stress_shortfall_count,
                "Safety Passed": "✅ Yes" if c.safety_check_passed else "❌ Rejected",
                "Violations": "; ".join(c.safety_violations) if c.safety_violations else "None",
            })
        st.dataframe(pd.DataFrame(cand_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No candidate updates currently staged.")

    # Candidate Promotion / Rejection Controls
    st.markdown("#### 🚀 Authorize Candidate Promotion")
    with st.form("promote_candidate_form"):
        pa, pb, pc = st.columns([1, 1, 1])
        with pa:
            cand_id = st.selectbox(
                "Select Candidate",
                [c.candidate_id for c in candidates if c.status.value == "SHADOW_MODE"] or ["No candidates ready"],
            )
        with pb:
            approver_role = st.selectbox("Authorization Role", ["HEAD_OF_TREASURY", "RISK_OFFICER", "AUDITOR"])
        with pc:
            approver_name = st.text_input("Approver Name", value="Chief Risk Officer")

        p_justification = st.text_input("Promotion Justification", value="Validated on 40-step chronological holdout with zero stress shortfalls.")
        promote_btn = st.form_submit_button("Promote to Production")

        if promote_btn and cand_id != "No candidates ready":
            success, msg = learning_service.promote_candidate(
                candidate_id=cand_id,
                approver=approver_name,
                approver_role=approver_role,
                justification=p_justification,
            )
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)

    # Propose New Candidate Update
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Propose Candidate System Update")
    with st.form("propose_candidate_form"):
        c_comp = st.selectbox("Subsystem Component", ["FORECASTING", "DECISION_OPTIMIZER", "SENTINEL_RISK", "SCENARIO_GENERATOR"])
        c_param = st.selectbox(
            "Target Parameter",
            [
                "ensemble_combining_weights",
                "volatility_stress_scenario_multiplier",
                "panic_sentinel_alert_threshold",
                "bypass_reserve_statutory_floor",  # Testing invariant rejection
            ],
        )
        c_val = st.text_input("Proposed Value (JSON or float)", value="0.75")
        c_desc = st.text_input("Update Description", value="Tighter alert threshold for high-velocity deposit shocks.")
        c_author = st.text_input("Author / Research Team", value="Liquidity Modeling Lab")
        
        prop_btn = st.form_submit_button("Submit Proposal for Safety Validation")
        if prop_btn:
            cand = learning_service.propose_update(
                component=c_comp,
                parameter_key=c_param,
                proposed_value=c_val,
                description=c_desc,
                author=c_author,
                holdout_sample_size=15,
                stress_shortfall_count=1 if "bypass_reserve" in c_param else 0,
            )
            if cand.safety_check_passed:
                st.success(f"Candidate {cand.candidate_id} passed safety invariants and moved to Shadow Mode!")
            else:
                st.error(f"Candidate {cand.candidate_id} REJECTED: {cand.rejection_reason}")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Audited Rollback Controls
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### ⏪ Audited Subsystem Rollback")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.85rem;'>Authorized rollback mechanism to immediately revert any degraded subsystem to a previous verified version.</p>",
        unsafe_allow_html=True,
    )

    with st.form("rollback_form"):
        ra, rb, rc = st.columns([1, 1, 1])
        with ra:
            rb_comp = st.selectbox("Subsystem to Roll Back", list(perf.active_versions.keys()))
        with rb:
            rb_ver = st.text_input("Target Revert Version", value="v1.0.0")
        with rc:
            rb_role = st.selectbox("Role for Rollback", ["HEAD_OF_TREASURY", "RISK_OFFICER", "TREASURY_ANALYST"])
        
        rb_just = st.text_input("Incident / Rollback Rationale", value="Emergency rollback due to unexpected slippage increase.")
        rb_btn = st.form_submit_button("Authorize and Execute Rollback")

        if rb_btn:
            success, msg = learning_service.rollback_component(
                component=rb_comp,
                target_version=rb_ver,
                author="Incident Commander",
                approver_role=rb_role,
                justification=rb_just,
            )
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)
