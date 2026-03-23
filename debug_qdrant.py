from app.rag.vectorstore import get_qdrant_client
from app.core.config import QDRANT_COLLECTION

client = get_qdrant_client()

points, next_offset = client.scroll(
    collection_name=QDRANT_COLLECTION,
    limit=10,
    with_payload=True,
    with_vectors=False,
)

print(f"Found {len(points)} points")
for i, p in enumerate(points, 1):
    print(f"\n--- Point {i} ---")
    print("ID:", p.id)
    print("Payload:", p.payload)