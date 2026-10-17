"""
Reserve Juggler: Autonomous Liquidity & Treasury Intelligence Platform.
Main Streamlit Application.
"""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

# Must be very first Streamlit command
st.set_page_config(
    page_title="Reserve Juggler — Treasury Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_this_dir = str(Path(__file__).resolve().parent)
_src_dir = str(Path(__file__).resolve().parent.parent.parent / "src")
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
for p in (_this_dir, _src_dir, _root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from views.decision import render_decision
    from views.decision_log import render_decision_log
    from views.forecasts import render_forecasts
    from views.learning import render_learning
    from views.overview import render_overview
    from views.portfolio import render_portfolio
    from views.sentinel import render_sentinel
    from views.stress_lab import render_stress_lab
    from styles import CUSTOM_CSS
except ImportError:
    from frontend.legacy_streamlit.views.decision import render_decision
    from frontend.legacy_streamlit.views.decision_log import render_decision_log
    from frontend.legacy_streamlit.views.forecasts import render_forecasts
    from frontend.legacy_streamlit.views.learning import render_learning
    from frontend.legacy_streamlit.views.overview import render_overview
    from frontend.legacy_streamlit.views.portfolio import render_portfolio
    from frontend.legacy_streamlit.views.sentinel import render_sentinel
    from frontend.legacy_streamlit.views.stress_lab import render_stress_lab
    from frontend.legacy_streamlit.styles import CUSTOM_CSS

from reserve_juggler.integration.facade import ApplicationFacade

# Inject Custom Dark Mode Glassmorphism Styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


import os

@st.cache_resource
def get_facade() -> ApplicationFacade:
    """Singleton application facade instance with configurable data source mode."""
    mode_env = os.environ.get("RESERVE_JUGGLER_MODE", "SYNTHETIC_DEMO").upper()
    use_synthetic = (mode_env != "PRODUCTION")
    db_path = os.environ.get("RESERVE_JUGGLER_DB_PATH", "reserve_juggler.db")
    return ApplicationFacade(persistence_file=db_path, use_synthetic=use_synthetic)


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

    # Data Source Banner
    ds_mode = snapshot.data_source_mode
    if ds_mode == "SYNTHETIC_DEMO":
        st.markdown(
            """
            <div style='background: rgba(99, 102, 241, 0.15); border: 1px solid #6366F1; border-radius: 6px; padding: 6px 12px; margin-bottom: 12px; font-size: 0.8rem; color: #C7D2FE; display: flex; justify-content: space-between;'>
                <span><b>Data Source:</b> SYNTHETIC_DEMO (Synthetic Demo Fixtures Active)</span>
                <span>Team 1 & 2 Modules: UNINTEGRATED (Stand-alone Mode)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif ds_mode == "LIVE":
        st.markdown(
            """
            <div style='background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 6px; padding: 6px 12px; margin-bottom: 12px; font-size: 0.8rem; color: #A7F3D0;'>
                <b>Data Source:</b> LIVE (Connected to Authoritative Production Services)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='background: rgba(239, 68, 68, 0.15); border: 1px solid #EF4444; border-radius: 6px; padding: 6px 12px; margin-bottom: 12px; font-size: 0.8rem; color: #FECACA;'>
                <b>Data Source:</b> {ds_mode} (Production Mode — Core Services Unintegrated)
            </div>
            """,
            unsafe_allow_html=True,
        )

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
