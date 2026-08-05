from dataclasses import dataclass

@dataclass
class Chunk:
    id: str
    document_name: str
    page_number: int
    text: str