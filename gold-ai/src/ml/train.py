import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit


FEATURES = [
    "return_1",
    "return_5",
    "ema_9",
    "ema_21",
    "ema_50",
    "atr",
    "rsi",
    "body",
    "range",
    "body_ratio",
    "ema_trend",
]


def create_target(df):

    data = df.copy()

    future = data["close"].shift(-12)

    data["target"] = 0

    data.loc[
        future >
        data["close"] * 1.001,
        "target"
    ] = 1

    data.loc[
        future <
        data["close"] * 0.999,
        "target"
    ] = -1

    return data


def train_model(df):

    data = create_target(df)

    data = data.dropna(
        subset=FEATURES + ["target"]
    )

    X = data[FEATURES]
    y = data["target"]

    split = int(
        len(data) * 0.8
    )

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    score = model.score(
        X_test,
        y_test
    )

    print(
        f"ML test accuracy: {score:.4f}"
    )

    joblib.dump(
        model,
        "models/gold_model.joblib"
    )

    print(
        "Model saved:"
        " models/gold_model.joblib"
    )

    return model