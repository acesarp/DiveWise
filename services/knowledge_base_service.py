from services.document_loader import DocumentLoader
from services.chunker import Chunker
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


class KnowledgeBaseService:

    def __init__(self):

        self.loader = DocumentLoader()
        self.chunker = Chunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.documents = []
        self.chunks = []

    def initialize(self):

        self.documents = self.loader.load_documents()
        # Database already exists
        if self.vector_store.count() > 0:
            return

        # First run
        self.documents = self.loader.load_documents()
        self.chunks = self.chunker.create_chunks(self.documents)
        embeddings = self.embedding_service.create_embeddings(self.chunks)
        self.vector_store.add_chunks(self.chunks, embeddings)

    def get_documents(self):
        return self.documents

    def get_chunk_count(self):
        return self.vector_store.count()

    def get_document_count(self):
        return len(self.documents)