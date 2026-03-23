import yfinance as yf

def get_news(ticker: str, limit: int = 5):
    tk = yf.Ticker(ticker)
    items = []

    try:
        news = getattr(tk, "news", [])
    except Exception:
        news = []

    for article in news[:limit]:
        if not isinstance(article, dict):
            continue

        content = article.get("content", {}) if isinstance(article.get("content"), dict) else {}
        items.append({
            "title": article.get("title"),
            "publisher": article.get("publisher"),
            "link": article.get("link"),
            "published": article.get("providerPublishTime"),
            "summary": content.get("summary"),
        })

    return items