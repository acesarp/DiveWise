from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


class RetrievalService:

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 5,   ):
        embedding = self.embedding_service.create_embedding(question)

        return self.vector_store.search(embedding=embedding, top_k=top_k,)