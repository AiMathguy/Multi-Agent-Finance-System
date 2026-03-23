import yfinance as yf

def get_company_snapshot(ticker: str):
    tk = yf.Ticker(ticker)
    info = tk.info or {}

    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "website": info.get("website"),
        "market_cap": info.get("marketCap"),
        "current_price": info.get("currentPrice"),
        "business_summary": info.get("longBusinessSummary"),
    }