import pandas as pd
from src.gann.engine import GannEngine


class DynamicGann:

    def __init__(
        self,
        swing_window=5,
        spacing=1.0,
        shape_num=8
    ):
        self.swing_window = swing_window
        self.spacing = spacing
        self.shape_num = shape_num

    def calculate(self, df):

        data = df.copy()

        highs = data["high"].rolling(
            self.swing_window * 2 + 1,
            center=True
        ).max()

        lows = data["low"].rolling(
            self.swing_window * 2 + 1,
            center=True
        ).min()

        data["swing_high"] = (
            data["high"] == highs
        )

        data["swing_low"] = (
            data["low"] == lows
        )

        swing_highs = data[
            data["swing_high"]
        ]

        swing_lows = data[
            data["swing_low"]
        ]

        if swing_highs.empty or swing_lows.empty:
            return None

        last_high = swing_highs.iloc[-1]
        last_low = swing_lows.iloc[-1]

        if last_low["datetime"] > last_high["datetime"]:

            direction = "UP"
            zero_price = float(
                last_low["low"]
            )
            zero_time = last_low["datetime"]

        else:

            direction = "DOWN"
            zero_price = float(
                last_high["high"]
            )
            zero_time = last_high["datetime"]

        gann = GannEngine(
            start_price=zero_price,
            direction=direction,
            spacing=self.spacing,
            total_degrees=360,
            bars_per_360=360,
            shape_num=self.shape_num,
        )

        levels = gann.generate_grid()

        return {
            "direction": direction,
            "zero_price": zero_price,
            "zero_time": zero_time,
            "levels": levels,
            "fan": gann.generate_fan(),
        }