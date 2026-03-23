import yfinance as yf
import pandas as pd
from app.utils.plotting import save_plot

def run_backtest(ticker: str):
    df = yf.download(ticker, period="5y")

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    df["Signal"] = (df["SMA20"] > df["SMA50"]).astype(int)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy"] = df["Returns"] * df["Signal"].shift(1)

    equity = (1 + df["Strategy"]).cumprod()

    plot_path = save_plot(df, ticker)

    return {
        "metrics": {
            "return": float((equity.iloc[-1] - 1) * 100)
        },
        "artifacts": {
            "plot": plot_path
        }
    }
