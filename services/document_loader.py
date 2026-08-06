from pathlib import Path

from pypdf import PdfReader

from models.document import Document
from services.text_cleaner import TextCleaner


class DocumentLoader:

    def __init__(self, documents_path: str = "data/documents"):
        self.documents_path = Path(documents_path)


    def load_documents(self) -> list[Document]:
        documents: list[Document] = []

        pdf_files = sorted(self.documents_path.glob("*.pdf"))

        for pdf_file in pdf_files:
            reader = PdfReader(pdf_file)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            clean_text = TextCleaner.clean(text)

            document = Document(file_name=pdf_file.name, file_path=str(pdf_file), page_count=len(reader.pages), text=clean_text)
            documents.append(document)

        return documents