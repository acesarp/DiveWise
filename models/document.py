from dataclasses import dataclass, field

@dataclass
class Document:
    file_name: str
    file_path: str
    page_count: int
    text: str
    page_texts: list[str] = field(default_factory=list)
