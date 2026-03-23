from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
from streamlit_option_menu import option_menu
import yfinance as yf


import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from app.ml.predictor import get_rf_backtest_data as _get_rf_backtest_data

@st.cache_data(show_spinner=False, ttl=600)
def get_rf_backtest_data_cached(ticker: str, period: str = "2y") -> pd.DataFrame:
    return _get_rf_backtest_data(ticker, period)

@st.cache_data(show_spinner=False, ttl=600)
def get_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    return df

API_URL = "http://127.0.0.1:8000/run"


result = st.session_state.get("last_result")

st.set_page_config(
    page_title="NeuralX Finance Intelligence Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Premium Investor-Grade CSS ----------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.14), transparent 24%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 20%),
            linear-gradient(180deg, #050b16 0%, #0a1220 100%);
        color: #e8eefc;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #e8eefc !important;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }

    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    [data-testid="stMain"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08111f 0%, #0c1627 100%) !important;
        border-right: 1px solid rgba(96, 165, 250, 0.12);
    }

    [data-testid="stSidebarContent"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e8eefc !important;
    }

    .hero-card {
        background:
            linear-gradient(135deg, rgba(17, 31, 55, 0.98), rgba(10, 20, 36, 0.98));
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 26px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow:
            0 18px 42px rgba(0,0,0,0.30),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #9fb0ca !important;
        font-size: 0.98rem;
    }

    .section-card {
        background:
            linear-gradient(180deg, rgba(11, 20, 34, 0.96), rgba(9, 17, 30, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 22px;
        padding: 1.1rem;
        margin-bottom: 1rem;
        box-shadow:
            0 10px 26px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .summary-card {
        background:
            linear-gradient(135deg, rgba(29, 78, 216, 0.20), rgba(14, 165, 233, 0.08));
        border: 1px solid rgba(96, 165, 250, 0.24);
        border-radius: 20px;
        padding: 1rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 26px rgba(0,0,0,0.20);
    }

    .metric-card {
        background:
            linear-gradient(180deg, rgba(15, 27, 45, 0.98), rgba(10, 18, 31, 0.98));
        border: 1px solid rgba(59, 130, 246, 0.16);
        border-radius: 20px;
        padding: 1rem;
        min-height: 118px;
        box-shadow:
            0 12px 28px rgba(0,0,0,0.24),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .metric-label {
        font-size: 0.76rem;
        color: #94a9c8 !important;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .metric-value {
        font-size: 1.58rem;
        font-weight: 800;
        color: #ffffff !important;
        line-height: 1.15;
    }

    .metric-sub {
        font-size: 0.84rem;
        color: #8ea2c2 !important;
        margin-top: 0.35rem;
    }

    .small-note {
        color: #9fb0ca !important;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .divider {
        border-top: 1px solid rgba(148, 163, 184, 0.12);
        margin: 1rem 0;
    }

    .pill {
        display: inline-block;
        padding: 0.34rem 0.74rem;
        margin: 0.18rem 0.3rem 0.18rem 0;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.22), rgba(14, 165, 233, 0.14));
        border: 1px solid rgba(96, 165, 250, 0.28);
        color: #dbeafe !important;
        font-size: 0.80rem;
        font-weight: 700;
    }

    .news-card {
        background:
            linear-gradient(180deg, rgba(10, 18, 31, 0.96), rgba(8, 15, 27, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow:
            0 8px 20px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .news-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0.3rem;
    }

    .news-meta {
        font-size: 0.80rem;
        color: #93a7c7 !important;
        margin-bottom: 0.45rem;
    }

    .status-good {
        color: #34d399 !important;
        font-weight: 700;
    }

    .status-warn {
        color: #fbbf24 !important;
        font-weight: 700;
    }

    .status-bad {
        color: #f87171 !important;
        font-weight: 700;
    }

    .stTextArea textarea {
        background: rgba(9, 17, 30, 0.98) !important;
        color: #e8eefc !important;
        border-radius: 16px !important;
        border: 1px solid rgba(96, 165, 250, 0.20) !important;
    }

    .stButton button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(135deg, #2563eb, #0ea5e9);
        color: white;
        border: none;
        font-weight: 800;
        padding: 0.76rem 1rem;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
    }

    .stButton button:hover {
        filter: brightness(1.06);
    }

    [data-testid="stSidebar"] .stButton button {
        text-align: left !important;
        justify-content: flex-start !important;
        background: rgba(9, 17, 30, 0.92) !important;
        border: 1px solid rgba(148, 163, 184, 0.10) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(37, 99, 235, 0.16) !important;
        border: 1px solid rgba(96, 165, 250, 0.22) !important;
    }

    [data-testid="stDataFrame"] {
        background: rgba(10, 18, 31, 0.96) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(148, 163, 184, 0.10) !important;
        overflow: hidden;
    }

    [data-testid="stJson"] {
        background: rgba(10, 18, 31, 0.96) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(148, 163, 184, 0.10) !important;
        padding: 0.4rem;
    }

    .footer-note {
        color: #8ca0c1 !important;
        font-size: 0.84rem;
        padding-top: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
def safe_get(d: Dict[str, Any], key: str, default: Any = "-") -> Any:
    if not isinstance(d, dict):
        return default
    value = d.get(key, default)
    return value if value not in [None, ""] else default


def fmt_num(v: Any) -> str:
    if v in [None, "", "-"]:
        return "-"
    try:
        if isinstance(v, (int, float)):
            if abs(v) >= 1_000_000_000_000:
                return f"{v / 1_000_000_000_000:.2f}T"
            if abs(v) >= 1_000_000_000:
                return f"{v / 1_000_000_000:.2f}B"
            if abs(v) >= 1_000_000:
                return f"{v / 1_000_000:.2f}M"
            return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
        return str(v)
    except Exception:
        return str(v)


def status_class(value: str) -> str:
    v = str(value).upper()
    if "NO_BUY" in v or "REJECT" in v:
        return "status-bad"
    if "PROPOSE_BUY" in v or "HOLD" in v or "REVIEW" in v:
        return "status-warn"
    if v == "BUY":
        return "status-good"
    return "status-warn"


def metric_card(label: str, value: Any, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_news(news: List[Dict[str, Any]]) -> None:
    if not news:
        st.info("No news available.")
        return

    for idx, item in enumerate(news, start=1):
        title = item.get("title") or (item.get("summary") or "Untitled article")[:120]
        publisher = item.get("publisher") or "Unknown publisher"
        published = item.get("published") or "Unknown date"
        summary = item.get("summary") or "No summary available."

        st.markdown(
            f"""
            <div class="news-card">
                <div class="news-title">{idx}. {title}</div>
                <div class="news-meta">{publisher} • {published}</div>
                <div>{summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom:10px;">
            <h2 style="margin-bottom:2px;">NeuralX</h2>
            <div class="small-note">
            Finance Intelligence Terminal
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    section = option_menu(
        menu_title=None,
        options=[
            "Executive",
            "News & RAG",
            "Prediction",
            "Risk",
            "Guardrails",
            "Raw",
        ],
        icons=[
            "speedometer2",
            "newspaper",
            "graph-up-arrow",
            "shield-check",
            "sliders",
            "braces",
        ],
        default_index=0,
        styles={
            "container": {"padding": "0"},
            "icon": {"color": "#7dd3fc"},
            "nav-link": {
                "font-size": "14px",
                "font-weight": "700",
                "padding": "12px 14px",
                "border-radius": "14px",
                "margin": "4px 0",
                "background-color": "rgba(15,23,42,0.7)",
                "color": "#dbeafe",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #2563eb, #0ea5e9)",
                "color": "white",
                "box-shadow": "0 10px 22px rgba(37,99,235,0.25)",
            },
        },
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    view_mode = option_menu(
        menu_title=None,
        options=["Presentation", "Detailed"],
        icons=["easel", "list-task"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0"},
            "nav-link": {
                "font-size": "12px",
                "padding": "8px",
                "border-radius": "10px",
                "background-color": "rgba(15,23,42,0.6)",
            },
            "nav-link-selected": {
                "background": "#2563eb",
                "color": "white",
            },
        },
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Quick Prompts")
    example_prompts = [
    "Analyze NVDA, include news, summarize recent themes",
    "Analyze NVDA, include news, run random forest prediction, add human intervention, and decide a risk-capped investment size",
    "Analyze AAPL, include news, run random forest prediction, and summarize risks",
    "Analyze MSFT, include news, run prediction, and apply risk policy",
]

    for i, ex in enumerate(example_prompts):
        if st.button(ex, key=f"quick_{i}"):
            st.session_state["prompt"] = ex

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### System Capabilities")
    st.markdown(
        """
        - RAG retrieval  
        - Predictive analytics  
        - Risk policy  
        - Human-in-the-loop  
        - Guardrails  
        """
    )


# ---------- Header ----------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📈 NeuralX Finance Intelligence Terminal</div>
        <div class="hero-subtitle">
            Investor-grade dashboard for finance research, predictive analytics, risk control, and human-guided decision support.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-card">
        <div class="metric-label">Workflow</div>
        <span class="pill">Prompt Parsing</span>
        <span class="pill">Company Snapshot</span>
        <span class="pill">News Retrieval</span>
        <span class="pill">RAG Reasoning</span>
        <span class="pill">Prediction</span>
        <span class="pill">Risk Control</span>
        <span class="pill">Human Intervention</span>
    </div>
    """,
    unsafe_allow_html=True,
)

default_prompt = st.session_state.get(
    "prompt",
    "Analyze NVDA, include news, run random forest prediction, add human intervention, and decide a risk-capped investment size",
)

prompt = st.text_area(
    "Enter a natural-language finance request",
    value=default_prompt,
    height=120,
    key="prompt",
)

col_run, col_hint = st.columns([1, 3])
with col_run:
    run_clicked = st.button("Run Analysis")
with col_hint:
    st.markdown(
        """
        <div class="section-card">
            <div class="small-note">
            Backend: <code>POST /run</code> at <code>http://127.0.0.1:8000</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if run_clicked:
    with st.spinner("Running finance pipeline..."):
        try:
            payload = {
    "prompt": "Analyze NVDA, include news, run random forest prediction, add human intervention, and decide a risk-capped investment size"
}
            st.write("Debug payload:", payload)
            st.write("Prompt from UI:", prompt)

            response = requests.post(
                API_URL,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=180,
            )
            st.write("Debug status:", response.status_code)

            if response.status_code != 200:
                st.error(f"API error: {response.status_code}")
                st.code(response.text)
            else:
                data = response.json()
                st.session_state["last_result"] = data

        except Exception as e:
            st.error(f"Failed to connect to API: {e}")


# ---------- Main ----------
result = st.session_state.get("last_result")

if result:
    snapshot = result.get("snapshot", {})
    metrics = result.get("metrics", {})
    prediction = metrics.get("prediction", {})
    risk = metrics.get("risk_decision", {})
    rag = result.get("rag", {})
    news = result.get("news", [])
    guardrails = result.get("guardrails", {})
    tasks = result.get("tasks", {})
    human = result.get("human_intervention", {})
    artifacts = result.get("artifacts", {})

    st.markdown("### System Overview")
    top = st.columns(5)
    with top[0]:
        metric_card("Ticker", safe_get(result, "ticker"), "Detected from prompt")
    with top[1]:
        metric_card("Current Price", fmt_num(safe_get(snapshot, "current_price")), "Latest snapshot")
    with top[2]:
        metric_card("Model Signal", safe_get(prediction, "latest_signal"), "Predictive output")
    with top[3]:
        metric_card("Risk Action", safe_get(risk, "action"), "Policy decision")
    with top[4]:
        metric_card("Human Approval", "Required" if human.get("required") else "No", "Execution gate")

    st.markdown(
        """
        <div class="section-card">
            <div class="metric-label">Detected capabilities in this run</div>
        """,
        unsafe_allow_html=True,
    )
    active_tasks = [k for k, v in tasks.items() if v]
    task_html = " ".join([f'<span class="pill">{k}</span>' for k in active_tasks]) if active_tasks else '<span class="pill">none</span>'
    st.markdown(task_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    def show_executive():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        if view_mode == "Presentation":
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="metric-label">Presentation Summary</div>
                    <div style="font-size:1.08rem; font-weight:800; color:white !important;">
                        {safe_get(snapshot, 'company_name')} • {safe_get(result, 'ticker')}
                    </div>
                    <div class="small-note" style="margin-top:0.5rem;">
                        The system retrieved current company context, reasoned over recent news, ran a predictive model,
                        applied a risk policy, and preserved a human approval gate before any buy proposal.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Price History")
        range_choice = st.selectbox(
            "Chart Range",
            ["6mo", "1y"],
            index=0,
            key="executive_chart_range",
        )

        price_df = get_price_history(result.get("ticker", "NVDA"), period=range_choice)

        if not price_df.empty and "Close" in price_df.columns:
            chart_df = price_df[["Close"]].copy()
            chart_df["SMA20"] = price_df["Close"].rolling(20).mean()
            chart_df["SMA50"] = price_df["Close"].rolling(50).mean()
            st.line_chart(chart_df, use_container_width=True)
        else:
            st.info("No chart data available.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1.1, 1.9])

        with c1:
            st.subheader("Company Snapshot")
            st.write(f"**Company:** {safe_get(snapshot, 'company_name')}")
            st.write(f"**Sector:** {safe_get(snapshot, 'sector')}")
            st.write(f"**Industry:** {safe_get(snapshot, 'industry')}")
            st.write(f"**Country:** {safe_get(snapshot, 'country')}")
            st.write(f"**Website:** {safe_get(snapshot, 'website')}")
            st.write(f"**Market Cap:** {fmt_num(safe_get(snapshot, 'market_cap'))}")
            st.write(f"**Current Price:** {fmt_num(safe_get(snapshot, 'current_price'))}")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.subheader("Recommendation Snapshot")
            signal_cls = status_class(safe_get(prediction, "latest_signal"))
            risk_cls = status_class(safe_get(risk, "action"))

            st.markdown(
                f"""
                <div><strong>Model Signal:</strong> <span class="{signal_cls}">{safe_get(prediction, "latest_signal")}</span></div>
                <div><strong>Risk Action:</strong> <span class="{risk_cls}">{safe_get(risk, "action")}</span></div>
                <div><strong>Human Approval:</strong> {"Required" if human.get("required") else "No"}</div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.subheader("Business Summary")
            st.write(safe_get(snapshot, "business_summary", "No business summary available."))

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.subheader("AI Executive Summary")
            st.write(safe_get(rag, "answer", "No RAG summary available."))

        st.markdown("</div>", unsafe_allow_html=True)

    def show_news():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Recent News")
        render_news(news)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("RAG Answer")
        st.write(safe_get(rag, "answer", "No answer available."))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Retrieved Sources")
        st.json(rag.get("sources", []))
        st.markdown("</div>", unsafe_allow_html=True)

    def show_prediction():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Predictive Analytics")

        if prediction:
            pcols = st.columns(4)
            with pcols[0]:
                metric_card("Model", safe_get(prediction, "model"), "Classifier")
            with pcols[1]:
                metric_card("Test Accuracy", safe_get(prediction, "test_accuracy"), "Held-out set")
            with pcols[2]:
                metric_card("Prob. Up", safe_get(prediction, "latest_probability_up"), "Latest confidence")
            with pcols[3]:
                metric_card("Latest Signal", safe_get(prediction, "latest_signal"), "Output")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            chart_tab1, chart_tab2, chart_tab3 = st.tabs(
                ["Price View", "Strategy vs Buy & Hold", "Feature Importance"]
            )

            with chart_tab1:
                st.subheader("Stock Price Trend")

                price_range = st.selectbox(
                    "Price Range",
                    ["6mo", "1y"],
                    index=0,
                    key="prediction_price_range"
                )

                price_df = get_price_history(result.get("ticker", "NVDA"), period=price_range)

                if not price_df.empty and "Close" in price_df.columns:
                    price_chart_df = price_df[["Close"]].copy()
                    price_chart_df["SMA20"] = price_df["Close"].rolling(20).mean()
                    price_chart_df["SMA50"] = price_df["Close"].rolling(50).mean()
                    st.line_chart(price_chart_df, use_container_width=True)
                else:
                    st.info("No price data available.")

            with chart_tab2:
                st.subheader("Random Forest Strategy vs Buy & Hold")

                bt_df = get_rf_backtest_data_cached(result.get("ticker", "NVDA"), period="2y")

                if not bt_df.empty:
                    perf_df = bt_df[["StrategyCurve", "BuyHoldCurve"]].copy()
                    st.line_chart(perf_df, use_container_width=True)

                    final_strategy = float(bt_df["StrategyCurve"].iloc[-1] - 1) * 100
                    final_bh = float(bt_df["BuyHoldCurve"].iloc[-1] - 1) * 100

                    b1, b2 = st.columns(2)
                    with b1:
                        metric_card("Strategy Return", f"{final_strategy:.2f}%", "RF long-only backtest")
                    with b2:
                        metric_card("Buy & Hold Return", f"{final_bh:.2f}%", "Benchmark")

                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.write("**Latest Model Signals**")
                    preview_cols = ["Close", "Prob_Up", "Pred", "StrategyCurve", "BuyHoldCurve"]
                    preview_df = bt_df[preview_cols].tail(10).copy()
                    st.dataframe(preview_df, use_container_width=True, hide_index=False)
                else:
                    st.info("Not enough data to build RF strategy backtest.")

            with chart_tab3:
                st.subheader("Feature Importance")

                fi = prediction.get("feature_importance", {})
                if fi:
                    df_fi = pd.DataFrame(
                        [{"feature": k, "importance": v} for k, v in fi.items()]
                    ).sort_values("importance", ascending=False)
                    st.bar_chart(df_fi.set_index("feature"))
                    st.dataframe(df_fi, use_container_width=True, hide_index=True)
                else:
                    st.info("No feature importance available.")
        else:
            st.info("No predictive output available.")

        st.markdown("</div>", unsafe_allow_html=True)

    def show_risk():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Risk Policy Decision")

        if risk:
            rcols = st.columns(4)
            with rcols[0]:
                metric_card("Action", safe_get(risk, "action"), "Policy result")
            with rcols[1]:
                metric_card("Shares", safe_get(risk, "shares"), "Proposed size")
            with rcols[2]:
                metric_card("Notional", fmt_num(safe_get(risk, "notional")), "Capital allocation")
            with rcols[3]:
                metric_card(
                    "Human Approval",
                    "Required" if risk.get("human_approval_required") else "No",
                    "Execution control",
                )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.subheader("Why the system chose this")
            reasons = risk.get("reasons", [])
            if reasons:
                for reason in reasons:
                    st.write(f"- {reason}")
            else:
                st.info("No risk explanation available.")
        else:
            st.info("No risk decision available.")

        st.markdown("</div>", unsafe_allow_html=True)

    def show_guardrails():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Guardrails & Control Plane")

        gcols = st.columns(4)
        with gcols[0]:
            metric_card("Max Steps", safe_get(guardrails, "max_steps"), "Pipeline cap")
        with gcols[1]:
            metric_card("Steps Used", safe_get(guardrails, "steps_used"), "Run usage")
        with gcols[2]:
            metric_card("Cost Cap USD", safe_get(guardrails, "cost_cap_usd"), "Budget ceiling")
        with gcols[3]:
            metric_card("Cost Used", safe_get(guardrails, "cost_used_estimate"), "Estimated")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Artifacts")
        st.json(artifacts if artifacts else {})

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Step Log")
        step_log = guardrails.get("step_log", [])
        if step_log:
            df_steps = pd.DataFrame(step_log)
            st.dataframe(df_steps, use_container_width=True, hide_index=True)
        else:
            st.info("No step log available.")

        st.markdown("</div>", unsafe_allow_html=True)

    def show_raw():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Full API Response")
        st.json(result)
        st.markdown("</div>", unsafe_allow_html=True)

    if section == "Executive":
        show_executive()
    elif section == "News & RAG":
        show_news()
    elif section == "Prediction":
        show_prediction()
    elif section == "Risk":
        show_risk()
    elif section == "Guardrails":
        show_guardrails()
    elif section == "Raw":
        show_raw()

else:
    st.markdown(
        """
        <div class="section-card">
            <h3 style="margin-top:0;">Investor-grade AI finance dashboard</h3>
            <div class="small-note">
                Run an analysis to demonstrate the full workflow:
                prompt → retrieval → RAG → prediction → risk control → human intervention.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer-note">
        NeuralX demo UI • Decision support only • Human approval required for any buy proposal
    </div>
    """,
    unsafe_allow_html=True,
)