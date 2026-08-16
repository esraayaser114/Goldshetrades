from src.data.historical import download_gold
from src.features.builder import build_features
from src.ml.train import train_model


df = download_gold(
    outputsize=5000
)

df = build_features(
    df
)

train_model(df)