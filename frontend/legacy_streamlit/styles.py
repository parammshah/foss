"""
Styling and Design System for Reserve Juggler Streamlit Dashboard.
Implements modern dark-mode aesthetic, glassmorphism surfaces, vivid metric cards,
and responsive status badges.
"""

CUSTOM_CSS = """
<style>
/* Modern typography import */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Glassmorphism card container */
.rj-card {
    background: rgba(18, 24, 38, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.rj-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
}

/* Metric tile */
.rj-metric-tile {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.rj-metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-bottom: 6px;
}

.rj-metric-val {
    font-size: 1.65rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    font-family: 'JetBrains Mono', monospace;
}

.rj-metric-sub {
    font-size: 0.78rem;
    color: #38BDF8;
    margin-top: 4px;
}

/* Status pills */
.badge-critical {
    background: rgba(239, 68, 68, 0.2);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    display: inline-block;
}

.badge-high {
    background: rgba(249, 115, 22, 0.2);
    color: #FB923C;
    border: 1px solid rgba(249, 115, 22, 0.4);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    display: inline-block;
}

.badge-medium {
    background: rgba(234, 179, 8, 0.2);
    color: #FACC15;
    border: 1px solid rgba(234, 179, 8, 0.4);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    display: inline-block;
}

.badge-ok {
    background: rgba(34, 197, 94, 0.2);
    color: #4ADE80;
    border: 1px solid rgba(34, 197, 94, 0.4);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    display: inline-block;
}

.badge-neutral {
    background: rgba(99, 102, 241, 0.2);
    color: #818CF8;
    border: 1px solid rgba(99, 102, 241, 0.4);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    display: inline-block;
}

/* Callout highlight */
.rj-callout {
    background: rgba(99, 102, 241, 0.08);
    border-left: 4px solid #6366F1;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.90rem;
    color: #E2E8F0;
}

/* Streamlit Header Override */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3 {
    font-weight: 700 !important;
    color: #F1F5F9 !important;
}
</style>
"""
