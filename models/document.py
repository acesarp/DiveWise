from dataclasses import dataclass

@dataclass
class Document:
    file_name: str
    file_path: str
    page_count: int
    text: str