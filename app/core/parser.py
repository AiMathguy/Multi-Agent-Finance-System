import re

def normalize_ticker(ticker: str) -> str:
    corrections = {"APPL": "AAPL"}
    return corrections.get(ticker.strip().upper(), ticker.strip().upper())

def extract_ticker(prompt: str):
    matches = re.findall(r"\b[A-Z]{1,5}\b", prompt.upper())
    blacklist = {
        "CSV", "ML", "AI", "SMA", "RSI", "MACD", "ROI",
        "NEWS", "PLOT", "AND", "WITH", "THE", "FOR", "RUN", "EXPORT", "RAG"
    }
    for m in matches:
        if m not in blacklist:
            return normalize_ticker(m)
    return None


def detect_tasks(prompt: str):
    p = prompt.lower()
    return {
        "report": any(x in p for x in ["report", "analyze", "summary"]),
        "news": "news" in p,
        "csv": "csv" in p or "export" in p,
        "backtest": "backtest" in p or "strategy" in p,
        "plot": "plot" in p or "chart" in p,
        "rag": any(x in p for x in ["why", "risk", "summarize", "theme", "what does", "insight", "research"]),
        "predictive": any(x in p for x in ["predict", "prediction", "random forest", "ml", "forecast"]),
    }