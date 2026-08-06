from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from models.chunk import Chunk


class VectorStore:
    COLLECTION_NAME = "divewise"

    def __init__(self, db_path: str = "data/chroma_db"):
        self.client = PersistentClient(path=db_path)
        self.collection: Collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME )


    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]    ) -> None:
        self.collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[   { "document": chunk.document_name, "page": chunk.page_number, }for chunk in chunks]
            )


    def search(self, embedding: list[float], top_k: int = 5   ):
        return self.collection.query(query_embeddings=[embedding], n_results=top_k)


    def count(self) -> int:
        return self.collection.count()


    def clear(self):
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(self.COLLECTION_NAME)