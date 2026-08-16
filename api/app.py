from fastapi import FastAPI

from src.data.historical import download_gold
from src.features.builder import build_features
from src.strategy.signal_engine import GoldSignalEngine


app = FastAPI(
    title="Gold Gann AI API",
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "name": "Gold Gann AI",
        "status": "online",
    }


@app.get("/signal")
def signal():

    df = download_gold(
        outputsize=500
    )

    features = build_features(
        df
    )

    strategy = GoldSignalEngine(
        max_stop=10.0,
        min_rr=3.0,
        max_targets=4,
    )

    result = strategy.generate(
        features
    )

    return result