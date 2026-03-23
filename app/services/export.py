import os
import pandas as pd
from app.core.config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_company_csv(snapshot: dict):
    path = os.path.join(OUTPUT_DIR, f"{snapshot['ticker']}_company.csv")
    pd.DataFrame([snapshot]).to_csv(path, index=False)
    return path

def save_news_csv(ticker: str, news_items: list):
    if not news_items:
        return None
    path = os.path.join(OUTPUT_DIR, f"{ticker}_news.csv")
    pd.DataFrame(news_items).to_csv(path, index=False)
    return path