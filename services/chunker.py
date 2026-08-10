from __future__ import annotations

from models import Chunk, Document
import uuid

class Chunker:

    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_CHUNK_OVERLAP = 150

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk overlap.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            # Keep page provenance so citations in the UI are meaningful.
            for page_number, text in enumerate(document.page_texts or [document.text], start=1):
                start = 0
                while start < len(text):
                    chunk_text = text[start:start + self.chunk_size]
                    chunks.append(Chunk(id=str(uuid.uuid4()),
                                            document_name=document.file_name,
                                            page_number=page_number,
                                            text=chunk_text)
                                            )
                    start += self.chunk_size - self.chunk_overlap

        return chunks
