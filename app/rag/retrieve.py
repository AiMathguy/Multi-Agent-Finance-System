from qdrant_client import models
from app.rag.vectorstore import get_vectorstore

def retrieve_company_context(query: str, ticker: str, k: int = 5):
    store = get_vectorstore()

    # 🔥 PRIORITY: NEWS FIRST
    news_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.ticker",
                match=models.MatchValue(value=ticker)
            ),
            models.FieldCondition(
                key="metadata.doc_type",
                match=models.MatchValue(value="news")
            )
        ]
    )

    docs = store.similarity_search(query, k=k, filter=news_filter)
    if docs:
        return docs

    # fallback to everything
    fallback_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.ticker",
                match=models.MatchValue(value=ticker)
            )
        ]
    )

    return store.similarity_search(query, k=k, filter=fallback_filter)