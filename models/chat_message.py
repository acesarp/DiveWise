from dataclasses import dataclass, field
from datetime import datetime

from models.search_result import SearchResult


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: list[SearchResult] = field(default_factory=list)