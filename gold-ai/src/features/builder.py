import numpy as np
import pandas as pd


def build_features(df):

    data = df.copy()

    # Returns
    data["return_1"] = (
        data["close"].pct_change()
    )

    data["return_5"] = (
        data["close"].pct_change(5)
    )

    # Moving averages
    data["ema_9"] = (
        data["close"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    data["ema_21"] = (
        data["close"]
        .ewm(span=21, adjust=False)
        .mean()
    )

    data["ema_50"] = (
        data["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    # ATR
    tr1 = (
        data["high"] -
        data["low"]
    )

    tr2 = (
        data["high"] -
        data["close"].shift()
    ).abs()

    tr3 = (
        data["low"] -
        data["close"].shift()
    ).abs()

    data["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    data["atr"] = (
        data["tr"]
        .rolling(14)
        .mean()
    )

    # RSI
    delta = data["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    data["rsi"] = (
        100 -
        (100 / (1 + rs))
    )

    # Candle structure
    data["body"] = (
        data["close"] -
        data["open"]
    )

    data["range"] = (
        data["high"] -
        data["low"]
    )

    data["body_ratio"] = (
        data["body"].abs() /
        data["range"].replace(0, np.nan)
    )

    # Trend
    data["ema_trend"] = np.where(
        data["ema_9"] >
        data["ema_21"],
        1,
        -1
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return data