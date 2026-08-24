import os
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from app.rag.embeddings import get_embeddings
from app.core.config import QDRANT_COLLECTION


_client = None


def get_qdrant_client():
    global _client
    if _client is not None:
        return _client

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if url and api_key:
        _client = QdrantClient(url=url, api_key=api_key)
    else:
        _client = QdrantClient(path="./qdrant_data")

    return _client


def ensure_collection():
    client = get_qdrant_client()

    try:
        collections = client.get_collections().collections
        existing = [c.name for c in collections]
    except Exception:
        existing = []

    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )

    for field_name in ["metadata.ticker", "metadata.doc_type", "metadata.source"]:
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def get_vectorstore():
    ensure_collection()
    client = get_qdrant_client()
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )