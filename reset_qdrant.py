from app.rag.vectorstore import get_qdrant_client
from app.core.config import QDRANT_COLLECTION

client = get_qdrant_client()

try:
    client.delete_collection(QDRANT_COLLECTION)
    print(f"Deleted collection: {QDRANT_COLLECTION}")
except Exception as e:
    print("Delete skipped:", e)