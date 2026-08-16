
from pathlib import Path

from src.data.historical import (
    download_gold
)

from src.strategy.signal_engine import (
    GoldSignalEngine
)

from src.backtest.full_backtest import (
    run_backtest,
    print_report,
)


def main():

    print("=" * 60)
    print("             GOLD AI SYSTEM")
    print("=" * 60)

    # ==========================
    # DATA
    # ==========================

    df = download_gold(
        outputsize=5000
    )

    print(
        f"\nLoaded candles: {len(df)}"
    )

    print(
        f"From: {df['datetime'].iloc[0]}"
    )

    print(
        f"To  : {df['datetime'].iloc[-1]}"
    )

    # ==========================
    # SIGNAL ENGINE
    # ==========================

    strategy = GoldSignalEngine(
        max_stop=10.0,
        min_rr=3.0,
        max_targets=4,
    )

    # ==========================
    # CURRENT SIGNAL
    # ==========================

    signal = strategy.generate(df)

    print("\n")
    print("=" * 60)
    print("CURRENT SIGNAL")
    print("=" * 60)

    for key, value in signal.items():
        print(
            f"{key:15}: {value}"
        )

    # ==========================
    # BACKTEST
    # ==========================

    print("\nRunning backtest...")

    results = run_backtest(
        df,
        strategy,
        max_trades_per_day=6
    )

    print_report(results)

    # ==========================
    # SAVE
    # ==========================

    if not results.empty:
        Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)
        

        results.to_csv(
            "data/processed/backtest_results.csv",
            index=False
        )

        print(
            "\nSaved:"
            " data/processed/backtest_results.csv"
        )


if __name__ == "__main__":
    main()