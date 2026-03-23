from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.vectorstore import get_vectorstore

def build_company_documents(ticker: str, snapshot: dict, news_items: list) -> List[Document]:
    docs = []

    company_text = f"""
Ticker: {snapshot.get('ticker')}
Company: {snapshot.get('company_name')}
Sector: {snapshot.get('sector')}
Industry: {snapshot.get('industry')}
Country: {snapshot.get('country')}
Website: {snapshot.get('website')}
Market Cap: {snapshot.get('market_cap')}
Current Price: {snapshot.get('current_price')}

Business Summary:
{snapshot.get('business_summary') or 'No business summary available.'}
""".strip()

    docs.append(
        Document(
            page_content=company_text,
            metadata={
                "ticker": ticker,
                "doc_type": "company_profile",
                "source": "yfinance_profile"
            }
        )
    )

    for item in news_items:
        news_text = f"""
Title: {item.get('title')}
Publisher: {item.get('publisher')}
Published: {item.get('published')}
Link: {item.get('link')}

Summary:
{item.get('summary') or 'No summary available.'}
""".strip()

        docs.append(
            Document(
                page_content=news_text,
                metadata={
                    "ticker": ticker,
                    "doc_type": "news",
                    "source": item.get("publisher") or "news",
                    "title": item.get("title")
                }
            )
        )

    return docs

def ingest_company_knowledge(ticker: str, snapshot: dict, news_items: list) -> int:
    docs = build_company_documents(ticker, snapshot, news_items)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(docs)
    store = get_vectorstore()
    store.add_documents(chunks)

    return len(chunks)