from pathlib import Path
import sys
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.historical import download_gold
from src.features.builder import build_features
from final_system import load_model, generate_signal, run_backtest


app = FastAPI(
    title="Gold Gann AI",
    description="Real-time XAU/USD AI + Gann Trading Signal API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = ROOT / "models" / "gold_model.joblib"
DASHBOARD_FILE = ROOT / "dashboard" / "index.html"


def get_data():

    df = download_gold(outputsize=5000)
    df = build_features(df)

    return df


def get_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Run python train.py"
        )

    return load_model()


@app.get("/")
def root():

    return {
        "name": "Gold Gann AI",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/dashboard"
    }


@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model": MODEL_FILE.exists(),
        "dashboard": DASHBOARD_FILE.exists()
    }


@app.get("/api/price")
def price():

    try:

        df = get_data()

        latest = df.iloc[-1]

        return {
            "symbol": "XAU/USD",
            "price": float(latest["close"]),
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "time": str(latest["datetime"])
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/signal")
def signal():

    try:

        df = get_data()
        model = get_model()

        result = generate_signal(
            df,
            model
        )

        latest = df.iloc[-1]

        result["symbol"] = "XAU/USD"
        result["latest_price"] = float(
            latest["close"]
        )
        result["latest_time"] = str(
            latest["datetime"]
        )

        return result

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/history")
def history(limit: int = 100):

    try:

        limit = min(
            max(limit, 1),
            500
        )

        df = get_data()

        data = df.tail(limit)

        candles = []

        for _, row in data.iterrows():

            candles.append({
                "time": str(row["datetime"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"])
            })

        return {
            "symbol": "XAU/USD",
            "interval": "5min",
            "count": len(candles),
            "candles": candles
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/backtest")
def backtest():

    try:

        df = get_data()
        model = get_model()

        results = run_backtest(
            df,
            model
        )

        if results.empty:

            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "net_r": 0,
                "profit_factor": 0
            }

        wins = int(
            (results["result"] == "WIN").sum()
        )

        losses = int(
            (results["result"] == "LOSS").sum()
        )

        trades = len(results)

        win_rate = (
            wins / trades * 100
            if trades
            else 0
        )

        net_r = float(
            results["r_multiple"].sum()
        )

        gross_profit = float(
            results.loc[
                results["r_multiple"] > 0,
                "r_multiple"
            ].sum()
        )

        gross_loss = abs(
            float(
                results.loc[
                    results["r_multiple"] < 0,
                    "r_multiple"
                ].sum()
            )
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0
        )

        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(
                win_rate,
                2
            ),
            "net_r": round(
                net_r,
                2
            ),
            "profit_factor": round(
                profit_factor,
                2
            )
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/dashboard")
def dashboard():

    if not DASHBOARD_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="Dashboard not found"
        )

    return FileResponse(
        DASHBOARD_FILE
    )