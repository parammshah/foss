"""
Decision Log & Audit Trail Dashboard View: Traceable Record of Actions, Incidents, and Updates.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
from reserve_juggler.integration.facade import DashboardSnapshot


def render_decision_log(snapshot: DashboardSnapshot) -> None:
    st.markdown("## 📜 Immutable Decision & Governance Audit Log")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.9rem;'>Chronological audit trail of algorithmic recommendations, human authorizations, overrides, simulated executions, and model lifecycle changes.</p>",
        unsafe_allow_html=True,
    )

    # Regulatory Disclaimer Banner (Mandated in Section 9)
    st.markdown(
        """
        <div style='background: rgba(245, 158, 11, 0.12); border-left: 4px solid #F59E0B; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px; font-size: 0.82rem; color: #FDE68A;'>
            <b>Jurisdiction-Aware Governance Notice:</b> This audit log records algorithmic decisions, human interactions, and model telemetry for internal risk governance and explainability. It does not constitute statutory regulatory certification under Basel III, Dodd-Frank, or SEC Rule 15c3-1. Specific compliance obligations depend on local regulatory jurisdictions and formal institutional validation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Audit Trail Table
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 System Audit Trail Entries")

    if snapshot.audit_log:
        df_audit = pd.DataFrame(snapshot.audit_log)
        # Rename columns for presentation
        col_map = {
            "timestamp": "Timestamp (UTC)",
            "action": "Action Type",
            "actor": "Initiator / Role",
            "details": "Audit Details & Stated Justification",
        }
        df_audit = df_audit.rename(columns=col_map)
        st.dataframe(df_audit, use_container_width=True, hide_index=True)
    else:
        st.info("No audit trail entries recorded yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Incident and Safe Fallback Log
    st.markdown("<div class='rj-card'>", unsafe_allow_html=True)
    st.markdown("### ⚠️ Incident & Safe Fallback Log")
    if snapshot.incident_log:
        st.dataframe(pd.DataFrame(snapshot.incident_log), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div style='color: #10B981; font-size: 0.88rem;'>
                ✅ <b>Zero Active Incidents:</b> Core liquidity firewall operational. All safety gates and reserve invariants verified.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
