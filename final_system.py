from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd

from src.data.historical import download_gold
from src.features.builder import build_features


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = Path("models/gold_model.joblib")
REPORT_DIR = Path("data/processed")

# ----------------------------
# Risk / Reward
# ----------------------------

MAX_STOP = 12.0

MIN_STOP = 1.0

ATR_STOP_MULTIPLIER = 2.0

MIN_RR = 2.0

MAX_TARGETS = 4

# ----------------------------
# Gann
# ----------------------------

GANN_SHAPE = 8
GANN_TOTAL_DEGREES = 360
GANN_SPACING = 1.0

LOOKBACK = 48

# ----------------------------
# Backtest
# ----------------------------

FUTURE_BARS = 24

MAX_TRADES_PER_DAY = 4

MIN_TRADES_PER_DAY = 1

# ----------------------------
# ML
# ----------------------------

MIN_CONFIDENCE = 0.62

# Stronger filtering
MIN_ENTRY_SCORE = 5


FEATURES = [
    "return_1",
    "return_5",
    "ema_9",
    "ema_21",
    "ema_50",
    "atr",
    "rsi",
    "body",
    "range",
    "body_ratio",
    "ema_trend",
]


# ============================================================
# UTILITIES
# ============================================================

def ensure_dirs():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    Path("models").mkdir(
        parents=True,
        exist_ok=True
    )


def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "ML model not found. Run: python train.py"
        )

    return joblib.load(MODEL_FILE)


def safe_float(x):

    try:
        value = float(x)

        if np.isfinite(value):
            return value

        return np.nan

    except Exception:

        return np.nan


# ============================================================
# GANN
# ============================================================

def gann_price(
    base,
    angle,
    direction,
    spacing=1.0
):

    scaled_base = base / spacing

    if scaled_base <= 0:
        return np.nan

    step = angle / 180.0

    if direction == "UP":

        result = (
            math.sqrt(scaled_base) + step
        ) ** 2

    else:

        value = (
            math.sqrt(scaled_base) - step
        )

        if value <= 0:
            return np.nan

        result = value ** 2

    return result * spacing


def build_gann_levels(
    anchor_price,
    direction
):

    deg_major = 360.0 / GANN_SHAPE

    deg_sub = deg_major / 2.0

    deg_minor = deg_sub / 2.0

    levels = []

    angle = 0.0

    while angle <= GANN_TOTAL_DEGREES + 1e-9:

        price = gann_price(
            anchor_price,
            angle,
            direction,
            GANN_SPACING
        )

        if pd.notna(price):

            if abs(angle % deg_major) < 1e-9:

                level_type = "MAJOR"

            elif abs(angle % deg_sub) < 1e-9:

                level_type = "SUB"

            else:

                level_type = "MINOR"

            levels.append({

                "angle": angle,

                "price": price,

                "type": level_type

            })

        angle += deg_minor

    return pd.DataFrame(levels)


def get_gann_context(
    df,
    index
):

    start = max(
        0,
        index - LOOKBACK + 1
    )

    window = df.iloc[
        start:index + 1
    ]

    if len(window) < 20:

        return None

    latest = window.iloc[-1]

    high_idx = window[
        "high"
    ].idxmax()

    low_idx = window[
        "low"
    ].idxmin()

    high_row = df.loc[
        high_idx
    ]

    low_row = df.loc[
        low_idx
    ]

    ema9 = safe_float(
        latest.get("ema_9")
    )

    ema21 = safe_float(
        latest.get("ema_21")
    )

    ema50 = safe_float(
        latest.get("ema_50")
    )

    # ----------------------------------------
    # Trend
    # ----------------------------------------

    if (
        np.isfinite(ema9)
        and
        np.isfinite(ema21)
        and
        np.isfinite(ema50)
        and
        ema9 > ema21 > ema50
    ):

        direction = "UP"

        anchor_price = float(
            low_row["low"]
        )

        anchor_time = low_row[
            "datetime"
        ]

    elif (
        np.isfinite(ema9)
        and
        np.isfinite(ema21)
        and
        np.isfinite(ema50)
        and
        ema9 < ema21 < ema50
    ):

        direction = "DOWN"

        anchor_price = float(
            high_row["high"]
        )

        anchor_time = high_row[
            "datetime"
        ]

    else:

        # No strong trend
        return None

    levels = build_gann_levels(
        anchor_price,
        direction
    )

    return {

        "direction": direction,

        "anchor_price": anchor_price,

        "anchor_time": anchor_time,

        "levels": levels

    }


def nearest_gann_level(
    price,
    levels
):

    if levels is None:
        return None

    if levels.empty:
        return None

    levels = levels.copy()

    levels["distance"] = (
        levels["price"] - price
    ).abs()

    row = levels.loc[
        levels["distance"].idxmin()
    ]

    return row


# ============================================================
# ML
# ============================================================

def ml_prediction(
    model,
    row
):

    X = pd.DataFrame(

        [[
            row.get(
                feature,
                np.nan
            )
            for feature in FEATURES
        ]],

        columns=FEATURES
    )

    if X.isna().any().any():

        return 0, 0.0

    prediction = int(
        model.predict(X)[0]
    )

    confidence = 0.0

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(X)[0]
        )

        classes = model.classes_

        if prediction in classes:

            position = list(
                classes
            ).index(
                prediction
            )

            confidence = float(
                probabilities[position]
            )

    return (
        prediction,
        confidence
    )


# ============================================================
# ENTRY FILTER
# ============================================================

def calculate_entry_score(
    row,
    direction,
    ml_signal,
    confidence
):

    score = 0

    reasons = []

    ema9 = safe_float(
        row.get("ema_9")
    )

    ema21 = safe_float(
        row.get("ema_21")
    )

    ema50 = safe_float(
        row.get("ema_50")
    )

    rsi = safe_float(
        row.get("rsi")
    )

    return1 = safe_float(
        row.get("return_1")
    )

    return5 = safe_float(
        row.get("return_5")
    )

    ema_trend = safe_float(
        row.get("ema_trend")
    )

    # ========================================================
    # 1. ML CONFIDENCE
    # ========================================================

    if confidence >= 0.70:

        score += 1

        reasons.append(
            "ML confidence strong"
        )

    elif confidence >= MIN_CONFIDENCE:

        score += 1

        reasons.append(
            "ML confidence acceptable"
        )

    # ========================================================
    # 2. EMA STRUCTURE
    # ========================================================

    if direction == "BUY":

        if (
            ema9 > ema21
            and
            ema21 > ema50
        ):

            score += 1

            reasons.append(
                "EMA bullish alignment"
            )

    else:

        if (
            ema9 < ema21
            and
            ema21 < ema50
        ):

            score += 1

            reasons.append(
                "EMA bearish alignment"
            )

    # ========================================================
    # 3. EMA TREND
    # ========================================================

    if direction == "BUY":

        if ema_trend > 0:

            score += 1

            reasons.append(
                "EMA trend bullish"
            )

    else:

        if ema_trend < 0:

            score += 1

            reasons.append(
                "EMA trend bearish"
            )

    # ========================================================
    # 4. MOMENTUM
    # ========================================================

    if direction == "BUY":

        if (
            return1 > 0
            and
            return5 > 0
        ):

            score += 1

            reasons.append(
                "Positive momentum"
            )

    else:

        if (
            return1 < 0
            and
            return5 < 0
        ):

            score += 1

            reasons.append(
                "Negative momentum"
            )

    # ========================================================
    # 5. RSI
    # ========================================================

    if direction == "BUY":

        if 50 <= rsi <= 68:

            score += 1

            reasons.append(
                "Healthy bullish RSI"
            )

    else:

        if 32 <= rsi <= 50:

            score += 1

            reasons.append(
                "Healthy bearish RSI"
            )

    # ========================================================
    # 6. ML DIRECTION
    # ========================================================

    expected_ml = (
        1
        if direction == "BUY"
        else 0
    )

    if ml_signal == expected_ml:

        score += 1

        reasons.append(
            "ML agrees"
        )

    return score, reasons


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    df,
    model,
    index=None
):

    if index is None:

        index = len(df) - 1

    if index < 60:

        return {

            "signal": "NO TRADE",

            "reason":
                "Not enough data"

        }

    row = df.iloc[index]

    price = safe_float(
        row["close"]
    )

    # ========================================================
    # ML
    # ========================================================

    ml_signal, confidence = (
        ml_prediction(
            model,
            row
        )
    )

    if confidence < MIN_CONFIDENCE:

        return {

            "signal": "NO TRADE",

            "reason":
                "ثقة النموذج منخفضة",

            "ml_signal":
                ml_signal,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "latest_price":
                round(
                    price,
                    5
                ),

            "latest_time":
                str(
                    row["datetime"]
                )

        }

    # ========================================================
    # GANN
    # ========================================================

    gann = get_gann_context(
        df,
        index
    )

    if gann is None:

        return {

            "signal": "NO TRADE",

            "reason":
                "السوق بدون اتجاه قوي",

            "ml_signal":
                ml_signal,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "latest_price":
                round(
                    price,
                    5
                ),

            "latest_time":
                str(
                    row["datetime"]
                )

        }

    direction = gann[
        "direction"
    ]

    # ========================================================
    # ML + GANN AGREEMENT
    # ========================================================

    expected_ml_signal = (

        1
        if direction == "UP"
        else 0

    )

    if ml_signal != expected_ml_signal:

        return {

            "signal": "NO TRADE",

            "reason":
                "النموذج والاتجاه مختلفان",

            "gann_direction":
                direction,

            "ml_signal":
                ml_signal,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "latest_price":
                round(
                    price,
                    5
                ),

            "latest_time":
                str(
                    row["datetime"]
                )

        }

    # ========================================================
    # SIGNAL DIRECTION
    # ========================================================

    trade_direction = (

        "BUY"
        if direction == "UP"
        else "SELL"

    )

    # ========================================================
    # ENTRY SCORE
    # ========================================================

    entry_score, reasons = (
        calculate_entry_score(

            row,

            trade_direction,

            ml_signal,

            confidence

        )
    )

    if entry_score < MIN_ENTRY_SCORE:

        return {

            "signal": "NO TRADE",

            "reason":
                "قوة الدخول غير كافية",

            "entry_score":
                entry_score,

            "required_score":
                MIN_ENTRY_SCORE,

            "gann_direction":
                direction,

            "ml_signal":
                ml_signal,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "latest_price":
                round(
                    price,
                    5
                ),

            "latest_time":
                str(
                    row["datetime"]
                )

        }

    # ========================================================
    # GANN LEVEL
    # ========================================================

    nearest = nearest_gann_level(

        price,

        gann[
            "levels"
        ]

    )

    if nearest is None:

        return {

            "signal": "NO TRADE",

            "reason":
                "لا يوجد مستوى Gann مناسب"

        }

    level_price = float(
        nearest["price"]
    )

    distance = abs(
        price - level_price
    )

    # ========================================================
    # ATR
    # ========================================================

    atr = safe_float(
        row["atr"]
    )

    if (
        not np.isfinite(atr)
        or
        atr <= 0
    ):

        return {

            "signal": "NO TRADE",

            "reason":
                "ATR غير صالح"

        }

    # ========================================================
    # DYNAMIC STOP
    # ========================================================

    risk = atr * ATR_STOP_MULTIPLIER

    risk = max(
        risk,
        MIN_STOP
    )

    risk = min(
        risk,
        MAX_STOP
    )

    entry = price

    # ========================================================
    # BUY
    # ========================================================

    if trade_direction == "BUY":

        stop = entry - risk

        targets = [

            entry + risk * 2,

            entry + risk * 3,

            entry + risk * 4,

            entry + risk * 5

        ]

    # ========================================================
    # SELL
    # ========================================================

    else:

        stop = entry + risk

        targets = [

            entry - risk * 2,

            entry - risk * 3,

            entry - risk * 4,

            entry - risk * 5

        ]

    targets = targets[
        :MAX_TARGETS
    ]

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "signal":
            trade_direction,

        "entry":
            round(
                entry,
                5
            ),

        "stop":
            round(
                stop,
                5
            ),

        "risk":
            round(
                risk,
                5
            ),

        "targets": [

            round(
                target,
                5
            )

            for target in targets

        ],

        "rr":
            2.0,

        "confidence":
            round(
                confidence,
                4
            ),

        "entry_score":
            entry_score,

        "entry_score_required":
            MIN_ENTRY_SCORE,

        "entry_reasons":
            reasons,

        "gann_direction":
            direction,

        "gann_angle":
            float(
                nearest["angle"]
            ),

        "gann_level":
            round(
                level_price,
                5
            ),

        "gann_distance":
            round(
                distance,
                5
            ),

        "anchor_price":
            round(
                gann["anchor_price"],
                5
            ),

        "anchor_time":
            str(
                gann["anchor_time"]
            ),

        "latest_price":
            round(
                price,
                5
            ),

        "latest_time":
            str(
                row["datetime"]
            ),

        "ema_trend":
            safe_float(
                row.get(
                    "ema_trend"
                )
            ),

        "rsi":
            round(
                safe_float(
                    row.get("rsi")
                ),
                2
            ),

        "atr":
            round(
                atr,
                5
            )

    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df,
    model
):

    trades = []

    daily_count = {}

    total = len(df)

    print(
        f"Backtest candles: {total}"
    )

    print(
        "Running..."
    )

    for i in range(
        60,
        total - FUTURE_BARS
    ):

        if (
            i - 60
        ) % 500 == 0:

            progress = (

                (i - 60)

                /

                max(
                    total
                    -
                    FUTURE_BARS
                    -
                    60,
                    1
                )

                * 100

            )

            print(
                f"Progress: {progress:.1f}%"
            )

        signal = generate_signal(

            df,

            model,

            i

        )

        direction = signal.get(
            "signal"
        )

        if direction not in (
            "BUY",
            "SELL"
        ):

            continue

        timestamp = pd.Timestamp(

            df.iloc[i][
                "datetime"
            ]

        )

        day = str(
            timestamp.date()
        )

        daily_count.setdefault(
            day,
            0
        )

        if (
            daily_count[day]
            >=
            MAX_TRADES_PER_DAY
        ):

            continue

        daily_count[day] += 1

        entry = float(
            signal["entry"]
        )

        stop = float(
            signal["stop"]
        )

        target = float(
            signal["targets"][0]
        )

        future = df.iloc[

            i + 1:

            i + 1 + FUTURE_BARS

        ]

        if future.empty:

            continue

        result = "OPEN"

        exit_price = float(
            future.iloc[-1]["close"]
        )

        exit_time = (
            future.iloc[-1][
                "datetime"
            ]
        )

        # ====================================================
        # BUY
        # ====================================================

        if direction == "BUY":

            for _, candle in (
                future.iterrows()
            ):

                candle_low = float(
                    candle["low"]
                )

                candle_high = float(
                    candle["high"]
                )

                # Stop first
                if candle_low <= stop:

                    result = "LOSS"

                    exit_price = stop

                    exit_time = (
                        candle[
                            "datetime"
                        ]
                    )

                    break

                if candle_high >= target:

                    result = "WIN"

                    exit_price = target

                    exit_time = (
                        candle[
                            "datetime"
                        ]
                    )

                    break

        # ====================================================
        # SELL
        # ====================================================

        else:

            for _, candle in (
                future.iterrows()
            ):

                candle_high = float(
                    candle["high"]
                )

                candle_low = float(
                    candle["low"]
                )

                # Stop first
                if candle_high >= stop:

                    result = "LOSS"

                    exit_price = stop

                    exit_time = (
                        candle[
                            "datetime"
                        ]
                    )

                    break

                if candle_low <= target:

                    result = "WIN"

                    exit_price = target

                    exit_time = (
                        candle[
                            "datetime"
                        ]
                    )

                    break

        # ====================================================
        # R MULTIPLE
        # ====================================================

        risk = abs(
            entry - stop
        )

        if result == "WIN":

            r_multiple = 2.0

        elif result == "LOSS":

            r_multiple = -1.0

        else:

            final_close = float(

                future.iloc[-1][
                    "close"
                ]

            )

            if direction == "BUY":

                move = (
                    final_close
                    -
                    entry
                )

            else:

                move = (
                    entry
                    -
                    final_close
                )

            r_multiple = (
                move
                /
                max(
                    risk,
                    1e-9
                )
            )

        trades.append({

            "entry_time":
                timestamp,

            "exit_time":
                exit_time,

            "direction":
                direction,

            "entry":
                entry,

            "stop":
                stop,

            "target":
                target,

            "risk":
                risk,

            "rr":
                2.0,

            "result":
                result,

            "r_multiple":
                r_multiple,

            "confidence":
                signal.get(
                    "confidence",
                    0.0
                ),

            "entry_score":
                signal.get(
                    "entry_score",
                    0
                ),

            "gann_angle":
                signal.get(
                    "gann_angle",
                    np.nan
                ),

            "gann_level":
                signal.get(
                    "gann_level",
                    np.nan
                ),

            "anchor_price":
                signal.get(
                    "anchor_price",
                    np.nan
                ),

            "rsi":
                signal.get(
                    "rsi",
                    np.nan
                ),

            "atr":
                signal.get(
                    "atr",
                    np.nan
                )

        })

    print(
        "Progress: 100.0%"
    )

    return pd.DataFrame(
        trades
    )


# ============================================================
# REPORT
# ============================================================

def report(results):

    print()

    print("=" * 60)

    print(
        "                 BACKTEST REPORT"
    )

    print("=" * 60)

    if results.empty:

        print(
            "No trades generated."
        )

        return

    total = len(results)

    wins = (
        results["result"]
        ==
        "WIN"
    ).sum()

    losses = (
        results["result"]
        ==
        "LOSS"
    ).sum()

    opens = (
        results["result"]
        ==
        "OPEN"
    ).sum()

    closed = wins + losses

    if closed > 0:

        win_rate = (
            wins
            /
            closed
            *
            100
        )

    else:

        win_rate = 0

    net_r = results[
        "r_multiple"
    ].sum()

    gross_profit = results.loc[

        results[
            "r_multiple"
        ] > 0,

        "r_multiple"

    ].sum()

    gross_loss = abs(

        results.loc[

            results[
                "r_multiple"
            ] < 0,

            "r_multiple"

        ].sum()

    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    else:

        profit_factor = np.inf

    daily = (

        results

        .assign(

            date=pd.to_datetime(

                results[
                    "entry_time"
                ]

            ).dt.date

        )

        .groupby(
            "date"
        )

        .size()

    )

    print(
        f"Total Trades  : {total}"
    )

    print(
        f"Closed Trades : {closed}"
    )

    print(
        f"Wins          : {wins}"
    )

    print(
        f"Losses        : {losses}"
    )

    print(
        f"Open          : {opens}"
    )

    print(
        f"Win Rate      : {win_rate:.2f}%"
    )

    print(
        f"Net R         : {net_r:.2f}R"
    )

    print(
        f"Profit Factor : {profit_factor:.2f}"
    )

    print(
        f"Avg trades/day: {daily.mean():.2f}"
    )

    print(
        f"Max trades/day: {daily.max()}"
    )

    print()

    results.to_csv(

        REPORT_DIR
        /
        "final_backtest.csv",

        index=False

    )

    daily.to_csv(

        REPORT_DIR
        /
        "daily_trades.csv"

    )

    print(
        "Saved:",
        REPORT_DIR
        /
        "final_backtest.csv"
    )

    print(
        "Saved:",
        REPORT_DIR
        /
        "daily_trades.csv"
    )


# ============================================================
# CURRENT SIGNAL
# ============================================================

def print_signal(
    signal
):

    print()

    print("=" * 60)

    print(
        "                 CURRENT SIGNAL"
    )

    print("=" * 60)

    for key, value in signal.items():

        if key == "targets":

            for i, target in enumerate(

                value,

                start=1

            ):

                print(

                    f"TP{i:<12}: "
                    f"{target}"

                )

        elif key == "entry_reasons":

            print(
                f"{key:<18}:"
            )

            for reason in value:

                print(
                    f"   - {reason}"
                )

        else:

            print(

                f"{key:<18}: "
                f"{value}"

            )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dirs()

    print("=" * 60)

    print(
        "                 GOLD GANN AI"
    )

    print("=" * 60)

    # ========================================================
    # DATA
    # ========================================================

    print()

    print(
        "1) Downloading Gold data..."
    )

    df = download_gold(
        outputsize=5000
    )

    # ========================================================
    # FEATURES
    # ========================================================

    print()

    print(
        "2) Building ML features..."
    )

    df = build_features(
        df
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()

    print(
        "3) Loading ML model..."
    )

    model = load_model()

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # CURRENT SIGNAL
    # ========================================================

    print()

    print(
        "4) Generating current signal..."
    )

    signal = generate_signal(

        df,

        model

    )

    print_signal(
        signal
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    print()

    print(
        "5) Running final backtest..."
    )

    results = run_backtest(

        df,

        model

    )

    report(
        results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 60)

    print(
        "                 SYSTEM SUMMARY"
    )

    print("=" * 60)

    print(
        "Current Signal :",
        signal.get(
            "signal",
            "NO TRADE"
        )
    )

    print(
        "Gold Price     :",
        signal.get(
            "latest_price",
            "-"
        )
    )

    print(
        "Gann Trend     :",
        signal.get(
            "gann_direction",
            "-"
        )
    )

    print(
        "ML Confidence  :",
        f"{signal.get('confidence', 0) * 100:.2f}%"
    )

    print(
        "Entry Score    :",
        signal.get(
            "entry_score",
            "-"
        )
    )

    if not results.empty:

        wins = (
            results["result"]
            ==
            "WIN"
        ).sum()

        losses = (
            results["result"]
            ==
            "LOSS"
        ).sum()

        closed = wins + losses

        if closed > 0:

            win_rate = (
                wins
                /
                closed
                *
                100
            )

        else:

            win_rate = 0

        net_r = results[
            "r_multiple"
        ].sum()

        print()

        print(
            "Backtest"
        )

        print(
            "Total Trades  :",
            len(results)
        )

        print(
            "Closed Trades :",
            closed
        )

        print(
            "Win Rate      :",
            f"{win_rate:.2f}%"
        )

        print(
            "Net R         :",
            f"{net_r:.2f}R"
        )

    print()

    print("=" * 60)

    print(
        "                 SYSTEM COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()