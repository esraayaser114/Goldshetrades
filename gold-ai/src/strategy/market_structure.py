import numpy as np
import pandas as pd


def find_swings(
    df,
    window=5
):

    df = df.copy()

    highs = df["high"].values
    lows = df["low"].values

    swing_high = np.zeros(len(df), dtype=bool)
    swing_low = np.zeros(len(df), dtype=bool)

    for i in range(window, len(df) - window):

        left_high = highs[
            i - window:i
        ]

        right_high = highs[
            i + 1:i + window + 1
        ]

        left_low = lows[
            i - window:i
        ]

        right_low = lows[
            i + 1:i + window + 1
        ]

        if (
            highs[i] > left_high.max()
            and highs[i] > right_high.max()
        ):
            swing_high[i] = True

        if (
            lows[i] < left_low.min()
            and lows[i] < right_low.min()
        ):
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low

    return df


def detect_trend(df):

    highs = df.loc[
        df["swing_high"],
        "high"
    ].tail(2).values

    lows = df.loc[
        df["swing_low"],
        "low"
    ].tail(2).values

    if len(highs) < 2 or len(lows) < 2:
        return "UNKNOWN"

    higher_high = highs[-1] > highs[-2]
    higher_low = lows[-1] > lows[-2]

    lower_high = highs[-1] < highs[-2]
    lower_low = lows[-1] < lows[-2]

    if higher_high and higher_low:
        return "UP"

    if lower_high and lower_low:
        return "DOWN"

    return "RANGE"