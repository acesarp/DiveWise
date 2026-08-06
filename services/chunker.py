from models.chunk import Chunk
from models.document import Document
import uuid

class Chunker:

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150

    def create_chunks(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            text = document.text

            start = 0

            while start < len(text):
                end = start + self.CHUNK_SIZE

                chunk_text = text[start:end]

                chunks.append(Chunk(id=str(uuid.uuid4()),
                                                    document_name=document.file_name,
                                                    page_number=0,
                                                    text=chunk_text)
                                        )

                start += self.CHUNK_SIZE - self.CHUNK_OVERLAP

        return chunks