from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Chunk:
    id: str
    document_name: str
    page_number: int
    text: str

@dataclass
class SearchResult:
    chunk: Chunk
    similarity: float


@dataclass
class ChatMessage:
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: list[SearchResult] = field(default_factory=list)


@dataclass
class Document:
    file_name: str
    file_path: str
    page_count: int
    text: str
    page_texts: list[str] = field(default_factory=list)

