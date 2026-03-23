from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import EMBED_MODEL

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)