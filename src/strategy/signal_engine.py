import pandas as pd

from src.strategy.market_structure import (
    find_swings,
    detect_trend,
)


class GoldSignalEngine:

    def __init__(
        self,
        max_stop=10.0,
        min_rr=3.0,
        max_targets=4,
    ):

        self.max_stop = max_stop
        self.min_rr = min_rr
        self.max_targets = max_targets

    def calculate_atr(
        self,
        df,
        period=14
    ):

        high_low = (
            df["high"] - df["low"]
        )

        high_close = (
            df["high"] -
            df["close"].shift()
        ).abs()

        low_close = (
            df["low"] -
            df["close"].shift()
        ).abs()

        tr = pd.concat(
            [
                high_low,
                high_close,
                low_close
            ],
            axis=1
        ).max(axis=1)

        return tr.rolling(period).mean()

    def generate(self, df):

        if len(df) < 100:
            return {
                "signal": "NO TRADE",
                "reason": "Not enough data",
            }

        data = find_swings(df)

        trend = detect_trend(data)

        if trend == "UNKNOWN":
            return {
                "signal": "NO TRADE",
                "reason": "Trend unavailable",
            }

        atr = self.calculate_atr(data).iloc[-1]

        if pd.isna(atr):
            return {
                "signal": "NO TRADE",
                "reason": "ATR unavailable",
            }

        price = float(
            data["close"].iloc[-1]
        )

        last_swing_low = data.loc[
            data["swing_low"],
            "low"
        ].iloc[-1] if data["swing_low"].any() else None

        last_swing_high = data.loc[
            data["swing_high"],
            "high"
        ].iloc[-1] if data["swing_high"].any() else None

        if trend == "UP":

            entry = price

            if last_swing_low is None:
                return {
                    "signal": "NO TRADE",
                    "reason": "No swing low",
                }

            stop = last_swing_low

            risk = entry - stop

            if risk <= 0:
                return {
                    "signal": "NO TRADE",
                    "reason": "Invalid bullish risk",
                }

            if risk > self.max_stop:
                return {
                    "signal": "NO TRADE",
                    "reason": "Stop too large",
                }

            targets = [
                entry + risk * 3,
                entry + risk * 4,
                entry + risk * 5,
                entry + risk * 6,
            ]

            return {
                "signal": "BUY",
                "trend": trend,
                "entry": entry,
                "stop_loss": stop,
                "risk": risk,
                "rr": 3.0,
                "target_1": targets[0],
                "target_2": targets[1],
                "target_3": targets[2],
                "target_4": targets[3],
                "atr": atr,
            }

        if trend == "DOWN":

            entry = price

            if last_swing_high is None:
                return {
                    "signal": "NO TRADE",
                    "reason": "No swing high",
                }

            stop = last_swing_high

            risk = stop - entry

            if risk <= 0:
                return {
                    "signal": "NO TRADE",
                    "reason": "Invalid bearish risk",
                }

            if risk > self.max_stop:
                return {
                    "signal": "NO TRADE",
                    "reason": "Stop too large",
                }

            targets = [
                entry - risk * 3,
                entry - risk * 4,
                entry - risk * 5,
                entry - risk * 6,
            ]

            return {
                "signal": "SELL",
                "trend": trend,
                "entry": entry,
                "stop_loss": stop,
                "risk": risk,
                "rr": 3.0,
                "target_1": targets[0],
                "target_2": targets[1],
                "target_3": targets[2],
                "target_4": targets[3],
                "atr": atr,
            }

        return {
            "signal": "NO TRADE",
            "reason": "Range market",
        }