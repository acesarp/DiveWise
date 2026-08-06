from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from models.chunk import Chunk
from models.document import Document
from models.search_result import SearchResult

class VectorStore:
    COLLECTION_NAME = "divewise"

    def __init__(self, db_path: str = "data/chroma_db"):
        self.client = PersistentClient(path=db_path)
        self.collection: Collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME )


    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]    ) -> None:
        self.collection.add(ids=[chunk.id for chunk in chunks],
                                                documents=[chunk.text for chunk in chunks],
                                                embeddings=embeddings,
                                                metadatas=[   { "document": chunk.document_name, "page": chunk.page_number, }for chunk in chunks]
                                    )


    def search(self, embedding: list[float], top_k: int = 5   ):
        result = self.collection.query(query_embeddings=[embedding],
                                                        n_results=top_k,
                                                        include=["documents", "metadatas", "distances"],
                                                    )
        
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            SearchResult(
                chunk=Chunk(
                    id=chunk_id,
                    document_name=metadata["document"],
                    page_number=int(metadata["page"]),
                    text=text,
                ),
                # Chroma distances are lower-is-better; expose a bounded score to the UI.
                similarity=1 / (1 + distance),
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
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(self.COLLECTION_NAME)
