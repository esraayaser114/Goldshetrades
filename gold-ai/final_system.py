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

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_ROOT / "models" / "gold_model.joblib"
REPORT_DIR = PROJECT_ROOT / "data" / "processed"

MAX_STOP = 10.0
MIN_RR = 3.0
MAX_TARGETS = 4

GANN_SHAPE = 8
GANN_TOTAL_DEGREES = 360
GANN_SPACING = 1.0

LOOKBACK = 48
FUTURE_BARS = 12

MAX_TRADES_PER_DAY = 6

MIN_CONFIDENCE = 0.55


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
    """
    Create required directories safely.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"ML model not found:\n{MODEL_FILE}\n\n"
            "Run this first:\n"
            "python train.py"
        )

    return joblib.load(MODEL_FILE)


def safe_float(x):

    try:
        value = float(x)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

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

            levels.append(
                {
                    "angle": angle,
                    "price": price,
                    "type": level_type
                }
            )

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

    if len(window) < 10:
        return None

    latest = window.iloc[-1]

    # -----------------------------------------
    # Causal swing selection
    # -----------------------------------------

    high_idx = window["high"].idxmax()
    low_idx = window["low"].idxmin()

    high_row = df.loc[high_idx]
    low_row = df.loc[low_idx]

    # -----------------------------------------
    # Trend direction
    # -----------------------------------------

    ema9 = safe_float(
        latest["ema_9"]
    )

    ema21 = safe_float(
        latest["ema_21"]
    )

    if not np.isfinite(ema9):
        return None

    if not np.isfinite(ema21):
        return None

    if ema9 > ema21:

        direction = "UP"

        anchor_price = float(
            low_row["low"]
        )

        anchor_time = low_row["datetime"]

    else:

        direction = "DOWN"

        anchor_price = float(
            high_row["high"]
        )

        anchor_time = high_row["datetime"]

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
        [
            [
                row.get(
                    feature,
                    np.nan
                )
                for feature in FEATURES
            ]
        ],
        columns=FEATURES
    )

    # -----------------------------------------
    # Missing values
    # -----------------------------------------

    if X.isna().any().any():

        return 0, 0.0

    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    prediction = int(
        model.predict(X)[0]
    )

    confidence = 0.0

    # -----------------------------------------
    # Probability
    # -----------------------------------------

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
            ).index(prediction)

            confidence = float(
                probabilities[position]
            )

    return prediction, confidence


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    df,
    model,
    index=None
):

    # -----------------------------------------
    # Index
    # -----------------------------------------

    if index is None:

        index = len(df) - 1

    if index < 60:

        return {
            "signal": "NO TRADE",
            "reason": "Not enough data"
        }

    row = df.iloc[index]

    price = safe_float(
        row["close"]
    )

    if not np.isfinite(price):

        return {
            "signal": "NO TRADE",
            "reason": "Invalid price"
        }

    # ========================================================
    # ML
    # ========================================================

    ml_signal, confidence = ml_prediction(
        model,
        row
    )

    # -----------------------------------------
    # Confidence filter
    # -----------------------------------------

    if confidence < MIN_CONFIDENCE:

        return {
            "signal": "NO TRADE",
            "reason": "Low ML confidence",
            "ml_signal": ml_signal,
            "confidence": confidence,
            "latest_price": price,
            "latest_time": str(
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
            "reason": "No Gann context",
            "ml_signal": ml_signal,
            "confidence": confidence,
            "latest_price": price,
            "latest_time": str(
                row["datetime"]
            )
        }

    # ========================================================
    # IMPORTANT:
    # ML classes:
    #
    # 1 = BUY
    # 0 = SELL / bearish
    #
    # Gann:
    # UP   -> 1
    # DOWN -> 0
    # ========================================================

    if gann["direction"] == "UP":

        expected_ml_signal = 1
        direction = 1

    else:

        expected_ml_signal = 0
        direction = -1

    # ========================================================
    # ML + GANN AGREEMENT
    # ========================================================

    if ml_signal != expected_ml_signal:

        return {
            "signal": "NO TRADE",
            "reason": "ML and Gann disagree",
            "gann_direction": gann["direction"],
            "ml_signal": ml_signal,
            "confidence": confidence,
            "latest_price": price,
            "latest_time": str(
                row["datetime"]
            )
        }

    # ========================================================
    # GANN LEVEL
    # ========================================================

    nearest = nearest_gann_level(
        price,
        gann["levels"]
    )

    if nearest is None:

        return {
            "signal": "NO TRADE",
            "reason": "No Gann level",
            "gann_direction": gann["direction"],
            "ml_signal": ml_signal,
            "confidence": confidence,
            "latest_price": price,
            "latest_time": str(
                row["datetime"]
            )
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

    if not np.isfinite(atr) or atr <= 0:

        return {
            "signal": "NO TRADE",
            "reason": "Invalid ATR",
            "gann_direction": gann["direction"],
            "ml_signal": ml_signal,
            "confidence": confidence,
            "latest_price": price,
            "latest_time": str(
                row["datetime"]
            )
        }

    # ========================================================
    # ENTRY
    # ========================================================

    entry = price

    # ATR based risk
    risk = min(
        max(
            atr * 0.8,
            0.5
        ),
        MAX_STOP
    )

    # ========================================================
    # BUY
    # ========================================================

    if direction == 1:

        stop = entry - risk

        targets = [
            entry + risk * 3,
            entry + risk * 4,
            entry + risk * 5,
            entry + risk * 6
        ]

        signal = "BUY"

    # ========================================================
    # SELL
    # ========================================================

    else:

        stop = entry + risk

        targets = [
            entry - risk * 3,
            entry - risk * 4,
            entry - risk * 5,
            entry - risk * 6
        ]

        signal = "SELL"

    targets = targets[:MAX_TARGETS]

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    return {

        "signal":
            signal,

        "entry":
            round(entry, 5),

        "stop":
            round(stop, 5),

        "risk":
            round(risk, 5),

        "targets":
            [
                round(
                    target,
                    5
                )
                for target in targets
            ],

        "rr":
            MIN_RR,

        "confidence":
            round(
                confidence,
                4
            ),

        "gann_direction":
            gann["direction"],

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
                row["ema_trend"]
            ),

        "rsi":
            round(
                safe_float(
                    row["rsi"]
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

    usable_total = max(
        total - FUTURE_BARS - 60,
        1
    )

    for i in range(
        60,
        total - FUTURE_BARS
    ):

        # -----------------------------------------
        # Progress
        # -----------------------------------------

        processed = i - 60

        if (
            processed % 500 == 0
            or i == total - FUTURE_BARS - 1
        ):

            progress = (
                processed
                / usable_total
                * 100
            )

            print(
                f"Progress: {progress:.1f}%"
            )

        # -----------------------------------------
        # Generate signal
        # -----------------------------------------

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
            df.iloc[i]["datetime"]
        )

        day = str(
            timestamp.date()
        )

        daily_count.setdefault(
            day,
            0
        )

        # -----------------------------------------
        # Daily trade limit
        # -----------------------------------------

        if (
            daily_count[day]
            >= MAX_TRADES_PER_DAY
        ):

            continue

        daily_count[day] += 1

        # -----------------------------------------
        # Trade values
        # -----------------------------------------

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

        exit_time = future.iloc[-1][
            "datetime"
        ]

        # ====================================================
        # BUY
        # ====================================================

        if direction == "BUY":

            for _, candle in future.iterrows():

                candle_low = float(
                    candle["low"]
                )

                candle_high = float(
                    candle["high"]
                )

                # Conservative:
                # stop first
                if candle_low <= stop:

                    result = "LOSS"

                    exit_price = stop

                    exit_time = candle[
                        "datetime"
                    ]

                    break

                if candle_high >= target:

                    result = "WIN"

                    exit_price = target

                    exit_time = candle[
                        "datetime"
                    ]

                    break

        # ====================================================
        # SELL
        # ====================================================

        else:

            for _, candle in future.iterrows():

                candle_high = float(
                    candle["high"]
                )

                candle_low = float(
                    candle["low"]
                )

                if candle_high >= stop:

                    result = "LOSS"

                    exit_price = stop

                    exit_time = candle[
                        "datetime"
                    ]

                    break

                if candle_low <= target:

                    result = "WIN"

                    exit_price = target

                    exit_time = candle[
                        "datetime"
                    ]

                    break

        # ====================================================
        # R MULTIPLE
        # ====================================================

        risk = abs(
            entry - stop
        )

        if result == "WIN":

            r_multiple = 3.0

        elif result == "LOSS":

            r_multiple = -1.0

        else:

            final_close = float(
                future.iloc[-1]["close"]
            )

            if direction == "BUY":

                move = (
                    final_close
                    - entry
                )

            else:

                move = (
                    entry
                    - final_close
                )

            r_multiple = (
                move
                / max(
                    risk,
                    1e-9
                )
            )

        # ====================================================
        # Save trade
        # ====================================================

        trades.append(
            {

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
                    MIN_RR,

                "result":
                    result,

                "exit_price":
                    exit_price,

                "r_multiple":
                    r_multiple,

                "confidence":
                    signal.get(
                        "confidence",
                        0.0
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
                    )
            }
        )

    print(
        "Progress: 100%"
    )

    return pd.DataFrame(
        trades
    )


# ============================================================
# REPORT
# ============================================================

def report(results):

    print()

    print(
        "=" * 60
    )

    print(
        "                 BACKTEST REPORT"
    )

    print(
        "=" * 60
    )

    if results.empty:

        print(
            "No trades generated."
        )

        return

    wins = (
        results["result"]
        == "WIN"
    ).sum()

    losses = (
        results["result"]
        == "LOSS"
    ).sum()

    opens = (
        results["result"]
        == "OPEN"
    ).sum()

    total = len(results)

    win_rate = (
        wins / total * 100
        if total > 0
        else 0
    )

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
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    daily = (
        results
        .assign(
            date=pd.to_datetime(
                results["entry_time"]
            ).dt.date
        )
        .groupby("date")
        .size()
    )

    print(
        f"Trades       : {total}"
    )

    print(
        f"Wins         : {wins}"
    )

    print(
        f"Losses       : {losses}"
    )

    print(
        f"Open         : {opens}"
    )

    print(
        f"Win Rate     : {win_rate:.2f}%"
    )

    print(
        f"Net R        : {net_r:.2f}R"
    )

    print(
        f"Profit Factor: {profit_factor:.2f}"
    )

    print(
        f"Avg trades/day: {daily.mean():.2f}"
    )

    print(
        f"Max trades/day: {daily.max()}"
    )

    # ========================================================
    # Save reports
    # ========================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    backtest_file = (
        REPORT_DIR
        / "final_backtest.csv"
    )

    daily_file = (
        REPORT_DIR
        / "daily_trades.csv"
    )

    results.to_csv(
        backtest_file,
        index=False
    )

    daily.to_csv(
        daily_file
    )

    print()

    print(
        f"Saved: {backtest_file}"
    )

    print(
        f"Saved: {daily_file}"
    )


# ============================================================
# CURRENT SIGNAL
# ============================================================

def print_signal(
    signal
):

    print()

    print(
        "=" * 60
    )

    print(
        "                 CURRENT SIGNAL"
    )

    print(
        "=" * 60
    )

    if not signal:

        print(
            "No signal data."
        )

        return

    for key, value in signal.items():

        if key == "targets":

            for i, target in enumerate(
                value,
                start=1
            ):

                print(
                    f"TP{i:<12}: {target}"
                )

        else:

            print(
                f"{key:<18}: {value}"
            )


# ============================================================
# SYSTEM SUMMARY
# ============================================================

def print_system_summary(
    signal,
    results
):

    print()

    print(
        "=" * 60
    )

    print(
        "                 SYSTEM SUMMARY"
    )

    print(
        "=" * 60
    )

    # Current signal
    current = signal.get(
        "signal",
        "UNKNOWN"
    )

    confidence = signal.get(
        "confidence",
        0.0
    )

    gann_direction = signal.get(
        "gann_direction",
        "-"
    )

    price = signal.get(
        "latest_price",
        "-"
    )

    print(
        f"Current Signal : {current}"
    )

    print(
        f"Gold Price     : {price}"
    )

    print(
        f"Gann Trend     : {gann_direction}"
    )

    if isinstance(
        confidence,
        (int, float)
    ):

        print(
            f"ML Confidence  : {confidence * 100:.2f}%"
        )

    # Backtest
    if (
        results is not None
        and not results.empty
    ):

        wins = (
            results["result"]
            == "WIN"
        ).sum()

        losses = (
            results["result"]
            == "LOSS"
        ).sum()

        total = len(results)

        win_rate = (
            wins / total * 100
            if total
            else 0
        )

        net_r = results[
            "r_multiple"
        ].sum()

        print()

        print(
            "Backtest"
        )

        print(
            f"Trades         : {total}"
        )

        print(
            f"Win Rate       : {win_rate:.2f}%"
        )

        print(
            f"Net R          : {net_r:.2f}R"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dirs()

    print(
        "=" * 60
    )

    print(
        "                 GOLD GANN AI"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # 1. DATA
    # ========================================================

    print()

    print(
        "1) Downloading Gold data..."
    )

    df = download_gold(
        outputsize=5000
    )

    if df.empty:

        raise RuntimeError(
            "No market data downloaded."
        )

    # ========================================================
    # 2. FEATURES
    # ========================================================

    print()

    print(
        "2) Building ML features..."
    )

    df = build_features(
        df
    )

    if df.empty:

        raise RuntimeError(
            "Feature dataframe is empty."
        )

    # ========================================================
    # 3. MODEL
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
    # 4. CURRENT SIGNAL
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
    # 5. BACKTEST
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
    # 6. SUMMARY
    # ========================================================

    print_system_summary(
        signal,
        results
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "                 SYSTEM COMPLETE"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()