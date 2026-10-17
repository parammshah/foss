"""
Reserve Juggler: Autonomous Liquidity & Treasury Intelligence Platform.
Main Streamlit Application.
"""

from __future__ import annotations
import streamlit as st

# Must be very first Streamlit command
st.set_page_config(
    page_title="Reserve Juggler — Treasury Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from reserve_juggler.dashboard.styles import CUSTOM_CSS
from reserve_juggler.dashboard.views.decision import render_decision
from reserve_juggler.dashboard.views.decision_log import render_decision_log
from reserve_juggler.dashboard.views.forecasts import render_forecasts
from reserve_juggler.dashboard.views.learning import render_learning
from reserve_juggler.dashboard.views.overview import render_overview
from reserve_juggler.dashboard.views.portfolio import render_portfolio
from reserve_juggler.dashboard.views.sentinel import render_sentinel
from reserve_juggler.dashboard.views.stress_lab import render_stress_lab
from reserve_juggler.integration.facade import ApplicationFacade

# Inject Custom Dark Mode Glassmorphism Styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_facade() -> ApplicationFacade:
    """Singleton application facade instance."""
    return ApplicationFacade(use_synthetic=True)


def main() -> None:
    facade = get_facade()

    # Sidebar Navigation & Controls
    with st.sidebar:
        st.markdown(
            """
            <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 12px;'>
                <span style='font-size: 2rem;'>⚖️</span>
                <div>
                    <div style='font-weight: 800; font-size: 1.15rem; color: #F8FAFC;'>RESERVE JUGGLER</div>
                    <div style='font-size: 0.72rem; color: #818CF8; letter-spacing: 0.05em;'>INTELLIGENCE & OVERSIGHT</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        
        # Operational Scenario Selector
        st.markdown("### 🎛️ Operational Scenario")
        scenario_map = {
            "Normal Operations": "NORMAL",
            "Banking System Stress": "BANKING_STRESS",
            "Rate Hike Shock (+75 bps)": "RATE_HIKE_SHOCK",
        }
        selected_label = st.selectbox(
            "Simulate Market Conditions",
            list(scenario_map.keys()),
            index=0,
        )
        selected_scenario = scenario_map[selected_label]
        if selected_scenario != facade.current_scenario:
            facade.switch_scenario(selected_scenario)
            st.rerun()

        st.markdown("---")

        # Page Navigation
        st.markdown("### 🧭 Platform Modules")
        pages = [
            "🏛️ Overview",
            "📈 Forecasts and Reliability",
            "💼 Portfolio",
            "🎯 Next Best Action",
            "🔬 Stress Lab",
            "📡 Macro-Panic Sentinel",
            "🧠 System Learning",
            "📜 Decision Log",
        ]
        chosen_page = st.radio("Navigation", pages, label_visibility="collapsed")

        st.markdown("---")
        
        # System status badge
        st.markdown(
            """
            <div style='padding: 10px; background: rgba(30, 41, 59, 0.4); border-radius: 8px; font-size: 0.75rem; color: #94A3B8;'>
                <div><b>Team 3 Master Architecture:</b></div>
                <div>• Macro-Panic Sentinel v1.0</div>
                <div>• Global Self-Learning Loop</div>
                <div>• Explainable AI & Human Gate</div>
                <div>• Future-Protected Firewall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Fetch snapshot
    snapshot = facade.get_dashboard_snapshot()

    # Route page rendering
    if "Overview" in chosen_page:
        render_overview(snapshot)
    elif "Forecasts" in chosen_page:
        render_forecasts(snapshot)
    elif "Portfolio" in chosen_page:
        render_portfolio(snapshot)
    elif "Next Best Action" in chosen_page:
        render_decision(snapshot, facade)
    elif "Stress Lab" in chosen_page:
        render_stress_lab(snapshot, facade)
    elif "Sentinel" in chosen_page:
        render_sentinel(snapshot, facade)
    elif "Learning" in chosen_page:
        render_learning(snapshot, facade)
    elif "Decision Log" in chosen_page:
        render_decision_log(snapshot)


if __name__ == "__main__":
    main()
