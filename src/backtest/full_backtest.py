import pandas as pd

from src.strategy.market_structure import (
    find_swings,
    detect_trend,
)
from src.strategy.signal_engine import (
    GoldSignalEngine,
)


def run_backtest(
    df,
    signal_engine,
    max_trades_per_day=6,
):

    data = df.copy()

    # Calculate swings ONCE
    data = find_swings(
        data,
        window=5
    )

    # Calculate ATR ONCE
    data["atr"] = signal_engine.calculate_atr(
        data
    )

    trades = []

    daily_count = {}

    start_index = 200

    for i in range(
        start_index,
        len(data) - 1
    ):

        current = data.iloc[i]

        day = current["datetime"].date()

        daily_count.setdefault(
            day,
            0
        )

        if daily_count[day] >= max_trades_per_day:
            continue

        window = data.iloc[
            :i + 1
        ]

        trend = detect_trend(
            window
        )

        if trend not in [
            "UP",
            "DOWN"
        ]:
            continue

        price = float(
            current["close"]
        )

        if trend == "UP":

            previous_lows = window.loc[
                window["swing_low"],
                "low"
            ]

            if previous_lows.empty:
                continue

            stop = float(
                previous_lows.iloc[-1]
            )

            risk = price - stop

            if risk <= 0:
                continue

            if risk > signal_engine.max_stop:
                continue

            direction = "BUY"

            targets = [
                price + risk * 3,
                price + risk * 4,
                price + risk * 5,
                price + risk * 6,
            ]

        else:

            previous_highs = window.loc[
                window["swing_high"],
                "high"
            ]

            if previous_highs.empty:
                continue

            stop = float(
                previous_highs.iloc[-1]
            )

            risk = stop - price

            if risk <= 0:
                continue

            if risk > signal_engine.max_stop:
                continue

            direction = "SELL"

            targets = [
                price - risk * 3,
                price - risk * 4,
                price - risk * 5,
                price - risk * 6,
            ]

        result = "OPEN"
        exit_price = price
        target_hit = 0

        # Forward candles
        end = min(
            i + 200,
            len(data)
        )

        for j in range(
            i + 1,
            end
        ):

            candle = data.iloc[j]

            if direction == "BUY":

                # Conservative rule:
                # stop first if both are touched
                if candle["low"] <= stop:

                    result = "LOSS"
                    exit_price = stop
                    break

                for k, target in enumerate(
                    targets
                ):

                    if candle["high"] >= target:

                        result = "WIN"
                        target_hit = k + 1
                        exit_price = target
                        break

            else:

                if candle["high"] >= stop:

                    result = "LOSS"
                    exit_price = stop
                    break

                for k, target in enumerate(
                    targets
                ):

                    if candle["low"] <= target:

                        result = "WIN"
                        target_hit = k + 1
                        exit_price = target
                        break

            if result != "OPEN":
                break

        if result == "OPEN":
            continue

        reward = abs(
            exit_price - price
        )

        rr = reward / risk

        trades.append({
            "entry_time": current["datetime"],
            "direction": direction,
            "entry": price,
            "stop": stop,
            "exit": exit_price,
            "risk": risk,
            "reward": reward,
            "rr": rr,
            "target_hit": target_hit,
            "result": result,
        })

        daily_count[day] += 1

    return pd.DataFrame(
        trades
    )


def print_report(results):

    print()
    print("=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)

    if results.empty:

        print("No trades found.")
        return

    total = len(results)

    wins = int(
        (results["result"] == "WIN").sum()
    )

    losses = int(
        (results["result"] == "LOSS").sum()
    )

    win_rate = (
        wins / total * 100
    )

    avg_rr = results["rr"].mean()

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
        f"Win Rate     : {win_rate:.2f}%"
    )

    print(
        f"Average R:R  : {avg_rr:.2f}"
    )

    print()
    print("Target distribution:")

    print(
        results[
            "target_hit"
        ].value_counts().sort_index()
    )

    print()
    print("Last trades:")

    print(
        results.tail(20).to_string(
            index=False
        )
    )