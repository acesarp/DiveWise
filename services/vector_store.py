from __future__ import annotations

import numpy as np
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from models import Chunk, Document, SearchResult

class VectorStore:
    COLLECTION_NAME = "divewise"
    BATCH_SIZE = 1_000
    DB_NAME = "data/chroma_db"
    
    def __init__(
        self,
        db_path: str = DB_NAME,
        collection_name: str = COLLECTION_NAME,
        similarity_metric: str = "cosine",
    ):
        self.client = PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.similarity_metric = similarity_metric
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.similarity_metric},
        )

    def _distance_to_similarity(self, distance: float) -> float:
        if self.similarity_metric == "cosine":
            return max(0.0, min(1.0, 1 - distance))

        return 1 / (1 + distance)


    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        for start in range(0, len(chunks), self.BATCH_SIZE):
            batch_chunks = chunks[start:start + self.BATCH_SIZE]
            batch_embeddings = embeddings[start:start + self.BATCH_SIZE]
            self.collection.add(
                ids=[chunk.id for chunk in batch_chunks],
                documents=[chunk.text for chunk in batch_chunks],
                embeddings=np.array(batch_embeddings),
                metadatas=[
                    {"document": chunk.document_name, "page": chunk.page_number}
                    for chunk in batch_chunks
                ],
            )


    def search(self, embedding: list[float], top_k: int = 5   ):
        result = self.collection.query(query_embeddings=[embedding],
                                                        n_results=top_k,
                                                        include=["documents", "metadatas", "distances"],
                                                    )
        
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return [
            SearchResult(
                chunk=Chunk(chunk_id, str(metadata["document"]), int(metadata["page"]), text),
                # Chroma distances are lower-is-better; expose a bounded score to the UI.
                similarity=self._distance_to_similarity(distance),
            )
            for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
        ]


    def count(self) -> int:
        return self.collection.count()

    def get_document_names(self) -> set[str]:
        result = self.collection.get(include=["metadatas"])
        return {
            metadata["document"]
            for metadata in result["metadatas"]
            if metadata and "document" in metadata
        }

    def get_documents(self) -> list[Document]:
        result = self.collection.get(include=["documents", "metadatas"])
        grouped: dict[str, list[tuple[int, str]]] = {}

        for text, metadata in zip(result["documents"], result["metadatas"]):
            if text and metadata and "document" in metadata:
                grouped.setdefault(metadata["document"], []).append((int(metadata["page"]), text))

        return [
            Document(
                file_name=file_name,
                file_path="",
                page_count=max(page for page, _ in chunks),
                text="\n\n".join(text for _, text in chunks),
            )
            for file_name, chunks in sorted(grouped.items())
        ]


    def clear(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)
