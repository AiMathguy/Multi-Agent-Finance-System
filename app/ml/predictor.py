import os
import json
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from app.ml.features import add_features
from app.core.config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)



FEATURE_COLUMNS = [
    "Return_1d",
    "Return_5d",
    "Volatility_10d",
    "Price_vs_SMA10",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "RSI_14",
    "Volume_Change",
]

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

FEATURE_COLUMNS = [
    "Return_1d",
    "Return_5d",
    "Volatility_10d",
    "Price_vs_SMA10",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "RSI_14",
    "Volume_Change",
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Return_1d"] = out["Close"].pct_change()
    out["Return_5d"] = out["Close"].pct_change(5)
    out["Volatility_10d"] = out["Return_1d"].rolling(10).std()

    out["SMA_10"] = out["Close"].rolling(10).mean()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()

    out["Price_vs_SMA10"] = out["Close"] / out["SMA_10"] - 1
    out["Price_vs_SMA20"] = out["Close"] / out["SMA_20"] - 1
    out["Price_vs_SMA50"] = out["Close"] / out["SMA_50"] - 1

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    out["Volume_Change"] = out["Volume"].pct_change()

    # next-day target
    out["Target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)

    return out


def get_rf_backtest_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    feat = add_features(df).dropna().copy()
    if len(feat) < 120:
        return pd.DataFrame()

    X = feat[FEATURE_COLUMNS]
    y = feat["Target"]

    split_idx = int(len(feat) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    test_df = feat.iloc[split_idx:].copy()
    test_df["Pred"] = model.predict(X_test)
    test_df["Prob_Up"] = model.predict_proba(X_test)[:, 1]

    # Long only when model predicts up
    test_df["Position"] = test_df["Pred"]
    test_df["AssetReturn"] = test_df["Close"].pct_change().fillna(0)
    test_df["StrategyReturn"] = test_df["AssetReturn"] * test_df["Position"].shift(1).fillna(0)

    test_df["BuyHoldCurve"] = (1 + test_df["AssetReturn"]).cumprod()
    test_df["StrategyCurve"] = (1 + test_df["StrategyReturn"]).cumprod()

    return test_df

def download_price_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data found for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    return df.copy()


def run_random_forest_prediction(ticker: str, period: str = "2y") -> dict:
    raw = download_price_data(ticker, period=period)
    feat = add_features(raw).dropna().copy()

    if len(feat) < 120:
        raise ValueError(f"Not enough data to train model for {ticker}")

    X = feat[FEATURE_COLUMNS]
    y = feat["Target"]

    split_idx = int(len(feat) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)

    latest_row = X.iloc[[-1]]
    latest_prob_up = float(model.predict_proba(latest_row)[0][1])
    latest_signal = "BUY" if latest_prob_up >= 0.55 else "HOLD/NO_BUY"

    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))

    result = {
        "ticker": ticker,
        "model": "RandomForestClassifier",
        "period": period,
        "test_accuracy": round(float(acc), 4),
        "latest_probability_up": round(latest_prob_up, 4),
        "latest_signal": latest_signal,
        "feature_importance": importance,
        "test_rows": int(len(X_test)),
        "train_rows": int(len(X_train)),
    }

    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_rf_prediction.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return {
        "metrics": result,
        "artifacts": {
            "prediction_json": out_path
        }
    }