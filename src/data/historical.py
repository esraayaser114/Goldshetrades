import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from src.config import API_KEY, SYMBOL, TIMEFRAME


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "XAUUSD_5m.csv"


def download_gold(outputsize=5000):

    if not API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing from .env"
        )

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": TIMEFRAME,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON",
        "order": "ASC",
    }

    print("Downloading XAU/USD data...")

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise RuntimeError(
            f"Twelve Data error: {data}"
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError(
            "Twelve Data returned no candles."
        )

    # ==========================================
    # DATETIME
    # ==========================================

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True
    )

    # ==========================================
    # NUMERIC DATA
    # ==========================================

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # ==========================================
    # CLEAN
    # ==========================================

    df = df.dropna(
        subset=[
            "datetime",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    # Remove duplicate candles
    df = df.drop_duplicates(
        subset=["datetime"]
    )

    # Sort chronologically
    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # ==========================================
    # REMOVE FUTURE DATA
    # ==========================================

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    before_count = len(df)

    df = df[
        df["datetime"] <= now_utc
    ].copy()

    removed = (
        before_count - len(df)
    )

    if removed > 0:

        print(
            f"Removed future candles: {removed}"
        )

    # ==========================================
    # BASIC OHLC VALIDATION
    # ==========================================

    invalid = (
        (df["high"] < df["low"])
        |
        (df["high"] < df["open"])
        |
        (df["high"] < df["close"])
        |
        (df["low"] > df["open"])
        |
        (df["low"] > df["close"])
    )

    invalid_count = int(
        invalid.sum()
    )

    if invalid_count > 0:

        print(
            f"Removing invalid candles: "
            f"{invalid_count}"
        )

        df = df[
            ~invalid
        ].copy()

    # ==========================================
    # SAVE
    # ==========================================

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ==========================================
    # REPORT
    # ==========================================

    print()
    print("=" * 50)
    print("SUCCESS")
    print("=" * 50)

    print(
        f"Rows       : {len(df)}"
    )

    print(
        f"File       : {OUTPUT_FILE}"
    )

    print(
        f"Timezone   : UTC"
    )

    if not df.empty:

        print(
            f"First candle: "
            f"{df['datetime'].iloc[0]}"
        )

        print(
            f"Last candle : "
            f"{df['datetime'].iloc[-1]}"
        )

    print()
    print(df.head())

    print()
    print(df.tail())

    return df


if __name__ == "__main__":
    download_gold()