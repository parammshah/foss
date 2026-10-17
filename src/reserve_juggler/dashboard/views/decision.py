"""
Next Best Action View: Grounded Decision Explanation and Human-in-the-Loop Oversight.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
from reserve_juggler.integration.contracts import ApprovalStatus, UserRole
from reserve_juggler.integration.facade import ApplicationFacade, DashboardSnapshot


def render_decision(snapshot: DashboardSnapshot, facade: ApplicationFacade) -> None:
    st.markdown("## ⚡ Next Best Action & Explainable AI")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.9rem;'>Grounded mathematical explanation, hard-gate compliance checks, alternative cost comparisons, and human oversight authorization.</p>",
        unsafe_allow_html=True,
    )

    dec = snapshot.latest_decision
    exp = snapshot.decision_explanation

    # Top Status Pill
    status_color = "badge-ok" if dec.approval_status == ApprovalStatus.APPROVED.value else (
        "badge-critical" if dec.approval_status == ApprovalStatus.REJECTED.value else "badge-medium"
    )
    st.markdown(
        f"<div style='margin-bottom: 15px;'>Approval Workflow Status: <span class='{status_color}'>{dec.approval_status}</span></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
        st.markdown("### 🏆 Recommended Allocation Action")
        st.markdown(
            f"""
            <div style='font-size: 1.4rem; font-weight: 700; color: #6366F1;'>
                {dec.selected_action.replace('_', ' ').title()}
            </div>
            <div style='font-family: monospace; font-size: 1.1rem; color: #F8FAFC; margin-top: 4px;'>
                Recommended Amount: <b>${dec.amount:,.2f}</b> on <code>{dec.target_asset}</code>
            </div>
            <div style='margin-top: 10px; font-size: 0.88rem; color: #94A3B8;'>
                Preserves Protected Reserve: <b>${dec.reserve:,.2f}</b> | 
                Surplus Remaining: <b>${dec.eligible_surplus:,.2f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🔍 Grounded Rationale Breakdown")
        st.code(exp.grounded_narrative, language="text")

        if exp.disclosed_missing_information:
            st.markdown("#### ⚠️ Disclosed Limitations / Caveats")
            for caveat in exp.disclosed_missing_information:
                st.warning(caveat)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
        st.markdown("### 💰 Transaction Cost Breakdown")
        cb = exp.cost_breakdown
        st.markdown(
            f"""
            - **Financing / Interest Cost:** ${cb.interest_cost:,.2f}
            - **Estimated Tax Impact:** ${cb.tax_impact:,.2f}
            - **Liquidation Penalty:** ${cb.liquidation_penalty:,.2f}
            - **Spread & Market Slippage:** ${cb.spread_slippage:,.2f}
            - **Total Estimated Transaction Cost:** <b style='color: #F87171;'>${cb.total_cost:,.2f}</b>
            """
        )

        st.markdown("### 🛡️ Hard Safety Gates Evaluation")
        for gate in exp.hard_gates:
            icon = "✅" if gate.passed else "❌"
            badge = "badge-ok" if gate.passed else "badge-critical"
            st.markdown(
                f"""
                <div style='padding: 6px 10px; background: rgba(30, 41, 59, 0.4); border-radius: 6px; margin-bottom: 6px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='font-weight: 600; font-size: 0.84rem;'>{icon} {gate.gate_name}</span>
                        <span class='{badge}'>{'PASSED' if gate.passed else 'FAILED'}</span>
                    </div>
                    <div style='font-size: 0.76rem; color: #94A3B8; margin-top: 2px;'>{gate.reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Alternatives Comparison Table
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 🔄 Rejected Alternatives Comparison")
    if exp.alternatives_comparison:
        alt_rows = []
        for a in exp.alternatives_comparison:
            alt_rows.append({
                "Alternative Action": a.action_type,
                "Target Instrument": a.target_asset,
                "Amount": f"${a.amount:,.2f}",
                "Est. Cost": f"${a.estimated_total_cost:,.2f}",
                "Cost Delta vs Selected": f"+${a.cost_delta_vs_selected:,.2f}",
                "Settlement Horizon": f"{a.settlement_horizon_hours}h",
                "Rejection Rationale": a.rejection_reason,
            })
        st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No viable alternatives met hard safety gate constraints.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Human-in-the-Loop Oversight Form
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### ✍️ Human-in-the-Loop Oversight & Approval Workflow")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.85rem;'>Enforces role-based permissions. All executed actions run in simulated execution mode with comprehensive audit trail logging.</p>",
        unsafe_allow_html=True,
    )

    with st.form("human_action_form"):
        fa, fb, fc = st.columns([1, 1, 1])
        with fa:
            actor_name = st.text_input("Authorizer Name", value="Jane Doe, CFA")
        with fb:
            actor_role = st.selectbox(
                "Authorizer Role",
                [r.value for r in UserRole],
                index=2,  # HEAD_OF_TREASURY default
            )
        with fc:
            decision_action = st.selectbox(
                "Workflow Decision",
                ["APPROVE_RECOMMENDATION", "MODIFY_AMOUNT_OR_ASSET", "CANCEL_ACTION", "REJECT_DECISION"],
            )

        reason = st.text_area("Audit Justification / Stated Rationale", "Reviewed and verified against intraday clearing liquidity schedule.")
        
        # Modification fields if requested
        mod_amt: Optional[float] = None
        if decision_action == "MODIFY_AMOUNT_OR_ASSET":
            mod_amt = st.number_input("Modified Action Amount ($)", value=float(dec.amount), step=1_000_000.0)

        submitted = st.form_submit_button("Submit & Execute Simulated Transaction")

        if submitted:
            action_map = {
                "APPROVE_RECOMMENDATION": ApprovalStatus.APPROVED.value,
                "MODIFY_AMOUNT_OR_ASSET": ApprovalStatus.MODIFIED.value,
                "CANCEL_ACTION": ApprovalStatus.CANCELLED.value,
                "REJECT_DECISION": ApprovalStatus.REJECTED.value,
            }
            mapped_action = action_map[decision_action]
            
            success, msg = facade.submit_human_action(
                action_type=mapped_action,
                actor_name=actor_name,
                actor_role=actor_role,
                stated_reason=reason,
                modified_amount=mod_amt,
            )

            if success:
                st.success(f"Success: {msg}")
                st.rerun()
            else:
                st.error(f"Action Rejected: {msg}")
    st.markdown("</div>", unsafe_allow_html=True)
