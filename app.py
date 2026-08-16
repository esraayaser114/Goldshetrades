import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data.historical import download_gold
from src.features.builder import build_features

from final_system import (
    load_model,
    generate_signal,
    run_backtest,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="She Trades Gold ",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS - ARABIC / RTL
# ============================================================

st.markdown(
    """
    <style>

    /* Main direction */
    .stApp {
        direction: rtl;
        text-align: right;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }

    /* Headers */
    h1, h2, h3, h4 {
        direction: rtl;
        text-align: right;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        direction: rtl;
        text-align: right;
    }

    /* Cards */
    .gold-card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 18px;
        background: rgba(128,128,128,0.05);
    }

    .signal-buy {
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        border: 2px solid #22c55e;
        background: rgba(34,197,94,0.10);
        margin-bottom: 20px;
    }

    .signal-sell {
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        border: 2px solid #ef4444;
        background: rgba(239,68,68,0.10);
        margin-bottom: 20px;
    }

    .signal-none {
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        border: 2px solid #eab308;
        background: rgba(234,179,8,0.10);
        margin-bottom: 20px;
    }

    .signal-title {
        font-size: 38px;
        font-weight: 800;
        margin: 10px;
    }

    .signal-subtitle {
        font-size: 18px;
        opacity: 0.8;
    }

    .status-online {
        color: #22c55e;
        font-weight: 700;
    }

    .status-offline {
        color: #ef4444;
        font-weight: 700;
    }

    .small-text {
        font-size: 13px;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_update" not in st.session_state:
    st.session_state.last_update = None

if "data" not in st.session_state:
    st.session_state.data = None

if "signal" not in st.session_state:
    st.session_state.signal = None

if "backtest" not in st.session_state:
    st.session_state.backtest = None

if "model" not in st.session_state:
    st.session_state.model = None


# ============================================================
# FUNCTIONS
# ============================================================

@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(ttl=300)
def get_gold_data():

    df = download_gold(
        outputsize=5000
    )

    return df


def prepare_data():

    df = get_gold_data()

    df = build_features(df)

    return df


def create_chart(df, signal):

    chart_df = df.tail(200).copy()

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=chart_df["datetime"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="الذهب",
        )
    )

    # EMA 9
    if "ema_9" in chart_df.columns:

        fig.add_trace(
            go.Scatter(
                x=chart_df["datetime"],
                y=chart_df["ema_9"],
                name="EMA 9",
                mode="lines",
            )
        )

    # EMA 21
    if "ema_21" in chart_df.columns:

        fig.add_trace(
            go.Scatter(
                x=chart_df["datetime"],
                y=chart_df["ema_21"],
                name="EMA 21",
                mode="lines",
            )
        )

    # EMA 50
    if "ema_50" in chart_df.columns:

        fig.add_trace(
            go.Scatter(
                x=chart_df["datetime"],
                y=chart_df["ema_50"],
                name="EMA 50",
                mode="lines",
            )
        )

    # Signal levels
    if signal:

        signal_type = signal.get("signal")

        if signal_type in ["BUY", "SELL"]:

            entry = signal.get("entry")
            stop = signal.get("stop")
            targets = signal.get("targets", [])

            if entry is not None:

                fig.add_hline(
                    y=entry,
                    annotation_text="الدخول",
                    line_dash="solid",
                )

            if stop is not None:

                fig.add_hline(
                    y=stop,
                    annotation_text="وقف الخسارة",
                    line_dash="dash",
                )

            for i, target in enumerate(
                targets,
                start=1
            ):

                fig.add_hline(
                    y=target,
                    annotation_text=f"الهدف {i}",
                    line_dash="dot",
                )

    fig.update_layout(

        height=650,

        xaxis_rangeslider_visible=False,

        template="plotly_dark",

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),

        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
    )

    return fig


def signal_card(signal):

    signal_type = signal.get(
        "signal",
        "NO TRADE"
    )

    if signal_type == "BUY":

        icon = "🟢"
        title = "شراء (BUY)"
        css = "signal-buy"

    elif signal_type == "SELL":

        icon = "🔴"
        title = "بيع (SELL)"
        css = "signal-sell"

    else:

        icon = "🟡"
        title = "لا توجد صفقة (NO TRADE)"
        css = "signal-none"

    reason = signal.get(
        "reason",
        "النظام لم يجد فرصة تداول مؤكدة."
    )

    st.markdown(
        f"""
        <div class="{css}">

            <div style="font-size:45px;">
                {icon}
            </div>

            <div class="signal-title">
                {title}
            </div>

            <div class="signal-subtitle">
                {reason}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ لوحة التحكم")

    st.markdown("---")

    auto_refresh = st.checkbox(
        "🔄 التحديث التلقائي",
        value=True
    )

    refresh_interval = st.slider(
        "مدة التحديث بالثواني",
        min_value=10,
        max_value=300,
        value=60,
        step=10,
    )

    st.markdown("---")

    st.subheader("🤖 حالة النظام")

    try:

        model = get_model()

        st.success(
            "نموذج ML محمل بنجاح"
        )

        st.session_state.model = model

    except Exception as e:

        st.error(
            f"فشل تحميل النموذج: {e}"
        )

    st.markdown("---")

    st.subheader("📡 مصدر البيانات")

    st.write("XAU/USD")
    st.write("شموع 5 دقائق")

    st.markdown("---")

    if st.button(
        "🔄 تحديث الآن",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.spinner(
        "جاري تحميل وتحليل بيانات الذهب..."
    ):

        df = prepare_data()

        model = get_model()

        signal = generate_signal(
            df,
            model
        )

        st.session_state.data = df
        st.session_state.signal = signal
        st.session_state.last_update = pd.Timestamp.now()

except Exception as e:

    st.error(
        f"حدث خطأ أثناء تشغيل النظام: {e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="text-align:center;">

    <h1>🟡 GOLD GANN AI</h1>

    <p style="font-size:18px; opacity:0.75;">
    نظام ذكي لتحليل الذهب في الوقت الحقيقي
    <br>
    Gann Analysis · Machine Learning · Automated Signals
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("---")


# ============================================================
# TOP METRICS
# ============================================================

latest = df.iloc[-1]

price = float(
    latest["close"]
)

signal_type = signal.get(
    "signal",
    "NO TRADE"
)

confidence = signal.get(
    "confidence",
    0
)

gann_direction = signal.get(
    "gann_direction",
    "-"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "💰 سعر الذهب",
        f"${price:,.2f}"
    )


with c2:

    if signal_type == "BUY":
        display_signal = "🟢 شراء"

    elif signal_type == "SELL":
        display_signal = "🔴 بيع"

    else:
        display_signal = "🟡 لا توجد صفقة"

    st.metric(
        "🎯 الإشارة الحالية",
        display_signal
    )


with c3:

    st.metric(
        "🧠 ثقة ML",
        f"{confidence * 100:.1f}%"
    )


with c4:

    direction_ar = {
        "UP": "صاعد",
        "DOWN": "هابط",
    }.get(
        gann_direction,
        "-"
    )

    st.metric(
        "📐 اتجاه Gann",
        direction_ar
    )


# ============================================================
# SIGNAL
# ============================================================

st.markdown(
    "## 🎯 إشارة التداول الحالية"
)

signal_card(signal)


# ============================================================
# TRADE DETAILS
# ============================================================

if signal_type in ["BUY", "SELL"]:

    st.markdown(
        "### 📌 تفاصيل الصفقة"
    )

    entry = signal.get(
        "entry"
    )

    stop = signal.get(
        "stop"
    )

    risk = signal.get(
        "risk"
    )

    rr = signal.get(
        "rr"
    )

    targets = signal.get(
        "targets",
        []
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "سعر الدخول",
            f"{entry:.5f}"
        )

    with c2:

        st.metric(
            "وقف الخسارة",
            f"{stop:.5f}"
        )

    with c3:

        st.metric(
            "المخاطرة",
            f"{risk:.5f}"
        )

    with c4:

        st.metric(
            "العائد / المخاطرة",
            f"1 : {rr:.1f}"
        )

    st.markdown(
        "#### 🎯 الأهداف"
    )

    target_columns = st.columns(
        len(targets)
    )

    for i, target in enumerate(
        targets
    ):

        with target_columns[i]:

            st.metric(
                f"الهدف {i + 1}",
                f"{target:.5f}"
            )


# ============================================================
# CHART
# ============================================================

st.markdown(
    "## 📈 الرسم البياني للذهب"
)

fig = create_chart(
    df,
    signal
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# MARKET INTELLIGENCE
# ============================================================

st.markdown(
    "## 🧠 تحليل السوق"
)


ema_trend = latest.get(
    "ema_trend",
    None
)

if ema_trend == -1:
    ema_ar = "هابط"
elif ema_trend == 1:
    ema_ar = "صاعد"
else:
    ema_ar = "-"


rsi = latest.get(
    "rsi",
    None
)

atr = latest.get(
    "atr",
    None
)

gann_level = signal.get(
    "gann_level",
    None
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "اتجاه EMA",
        ema_ar
    )


with c2:

    if pd.notna(rsi):

        st.metric(
            "RSI",
            f"{float(rsi):.2f}"
        )

    else:

        st.metric(
            "RSI",
            "-"
        )


with c3:

    if pd.notna(atr):

        st.metric(
            "ATR",
            f"{float(atr):.2f}"
        )

    else:

        st.metric(
            "ATR",
            "-"
        )


with c4:

    if gann_level is not None:

        st.metric(
            "مستوى Gann",
            f"{float(gann_level):,.2f}"
        )

    else:

        st.metric(
            "مستوى Gann",
            "-"
        )


# ============================================================
# BACKTEST
# ============================================================

st.markdown(
    "## 📊 أداء الاستراتيجية"
)

run_backtest_button = st.button(
    "▶️ تشغيل الاختبار التاريخي",
    use_container_width=True
)


if run_backtest_button:

    with st.spinner(
        "جاري تشغيل الاختبار التاريخي..."
    ):

        results = run_backtest(
            df,
            model
        )

        st.session_state.backtest = results


results = st.session_state.backtest


if results is not None and not results.empty:

    total = len(results)

    wins = (
        results["result"] == "WIN"
    ).sum()

    losses = (
        results["result"] == "LOSS"
    ).sum()

    open_trades = (
        results["result"] == "OPEN"
    ).sum()

    closed = wins + losses

    if closed > 0:

        win_rate = (
            wins / closed * 100
        )

    else:

        win_rate = 0

    net_r = results[
        "r_multiple"
    ].sum()

    gross_profit = results.loc[
        results["r_multiple"] > 0,
        "r_multiple"
    ].sum()

    gross_loss = abs(
        results.loc[
            results["r_multiple"] < 0,
            "r_multiple"
        ].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit / gross_loss
        )

    else:

        profit_factor = 0

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "إجمالي الصفقات",
            total
        )

    with c2:

        st.metric(
            "الصفقات المغلقة",
            closed
        )

    with c3:

        st.metric(
            "نسبة النجاح",
            f"{win_rate:.2f}%"
        )

    with c4:

        st.metric(
            "صافي R",
            f"{net_r:.2f}R"
        )

    with c5:

        st.metric(
            "Profit Factor",
            f"{profit_factor:.2f}"
        )

    st.markdown(
        f"""
        **الرابحة:** {wins}

        **الخاسرة:** {losses}

        **المفتوحة:** {open_trades}
        """
    )

    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    st.markdown(
        "### 📈 منحنى أداء الاستراتيجية"
    )

    equity = (
        results["r_multiple"]
        .cumsum()
    )

    equity_df = pd.DataFrame(
        {
            "رقم الصفقة":
                range(1, len(equity) + 1),

            "صافي R":
                equity.values,
        }
    )

    equity_fig = go.Figure()

    equity_fig.add_trace(
        go.Scatter(
            x=equity_df["رقم الصفقة"],
            y=equity_df["صافي R"],
            mode="lines",
            name="Equity Curve",
        )
    )

    equity_fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="رقم الصفقة",
        yaxis_title="صافي R",
    )

    st.plotly_chart(
        equity_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # LAST TRADES
    # --------------------------------------------------------

    st.markdown(
        "### 🧾 آخر الصفقات"
    )

    display_columns = [
        "entry_time",
        "direction",
        "entry",
        "stop",
        "target",
        "result",
        "r_multiple",
        "confidence",
    ]

    available_columns = [
        c
        for c in display_columns
        if c in results.columns
    ]

    recent = results[
        available_columns
    ].tail(15).copy()

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "اضغطي على «تشغيل الاختبار التاريخي» لعرض أداء الاستراتيجية."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

last_update = st.session_state.last_update

if last_update is not None:

    st.markdown(
        f"""
        <div style="text-align:center; opacity:0.65;">

        آخر تحديث:
        {last_update.strftime("%Y-%m-%d %H:%M:%S")}

        <br>

        Gold Gann AI · Version 1.0

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# AUTO REFRESH
# ============================================================

if auto_refresh:

    time.sleep(
        refresh_interval
    )

    st.rerun()