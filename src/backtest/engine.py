import pandas as pd


class Backtester:

    def __init__(self, signal_engine):

        self.signal_engine = signal_engine

    def run(self, df):

        trades = []

        for i in range(100, len(df) - 1):

            window = df.iloc[:i + 1]

            signal = self.signal_engine.generate(
                window
            )

            if signal["signal"] == "NO TRADE":
                continue

            next_candle = df.iloc[i + 1]

            entry = signal["entry"]
            stop = signal["stop_loss"]
            target = signal["target"]

            result = "OPEN"

            if signal["signal"] == "BUY":

                if next_candle["low"] <= stop:
                    result = "LOSS"

                elif next_candle["high"] >= target:
                    result = "WIN"

            elif signal["signal"] == "SELL":

                if next_candle["high"] >= stop:
                    result = "LOSS"

                elif next_candle["low"] <= target:
                    result = "WIN"

            trades.append({
                "datetime": next_candle["datetime"],
                "signal": signal["signal"],
                "entry": entry,
                "stop": stop,
                "target": target,
                "rr": signal["rr"],
                "result": result
            })

        return pd.DataFrame(trades)