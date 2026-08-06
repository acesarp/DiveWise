from models.chunk import Chunk
from models.document import Document
import uuid

class Chunker:

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150

    def create_chunks(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            # Keep page provenance so citations in the UI are meaningful.
            for page_number, text in enumerate(document.page_texts or [document.text], start=1):
                start = 0
                while start < len(text):
                    chunk_text = text[start:start + self.CHUNK_SIZE]
                    chunks.append(Chunk(id=str(uuid.uuid4()),
                                            document_name=document.file_name,
                                            page_number=page_number,
                                            text=chunk_text)
                                            )
                    start += self.CHUNK_SIZE - self.CHUNK_OVERLAP

        return chunks
