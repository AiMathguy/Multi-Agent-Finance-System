import pandas as pd


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

    # target: next-day direction
    out["Target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)

    return out