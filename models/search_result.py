from dataclasses import dataclass

from models.chunk import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    similarity: float