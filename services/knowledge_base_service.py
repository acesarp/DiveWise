from services.document_loader import DocumentLoader
from services.chunker import Chunker
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from dataclasses import dataclass
from collections.abc import Callable

@dataclass(frozen=True)
class KnowledgeBaseStatistics:
    documents: int
    pages: int
    chunks: int


class KnowledgeBaseService:

    def __init__(self, embedding_model: str = "text-embedding-3-small"):

        self.loader = DocumentLoader()
        self.chunker = Chunker()
        self.embedding_service = EmbeddingService(model=embedding_model)
        self.vector_store = VectorStore(collection_name=f"divewise-{embedding_model}")
        self.documents = []
        self.chunks = []

    def initialize(self, progress_callback: Callable[[int, int, str], None] | None = None):
        available_document_names = self.loader.get_document_names()
        indexed_document_names = self.vector_store.get_document_names()
        new_document_names = available_document_names - indexed_document_names

        # Add only newly discovered PDFs. Existing documents are neither cleared or re-embedded
        if new_document_names:
            documents_to_index = self.loader.load_documents(new_document_names, progress_callback=progress_callback)
            self.chunks = self.chunker.create_chunks(documents_to_index)
            embeddings = self.embedding_service.create_embeddings(self.chunks, progress_callback=progress_callback)
            self.vector_store.add_chunks(self.chunks, embeddings)

        self.documents = self.vector_store.get_documents()
        if progress_callback:
            progress_callback(1, 1, "Knowledge base ready")

    def get_documents(self):
        return self.documents

    def get_chunk_count(self):
        return self.vector_store.count()

    def get_document_count(self):
        return len(self.documents)

    def get_statistics(self) -> KnowledgeBaseStatistics:
        return KnowledgeBaseStatistics(self.get_document_count(), sum(document.page_count for document in self.documents), self.get_chunk_count())
