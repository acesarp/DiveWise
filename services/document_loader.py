from pathlib import Path

from pypdf import PdfReader

from models.document import Document
from services.text_cleaner import TextCleaner


class DocumentLoader:

    def __init__(self, documents_path: str = "data/documents"):
        self.documents_path = Path(documents_path)


    def get_document_names(self) -> set[str]:
        return {pdf_file.name for pdf_file in self.documents_path.glob("*.pdf")}

    def load_documents(self, file_names: set[str] | None = None) -> list[Document]:
        documents: list[Document] = []

        pdf_files = sorted(self.documents_path.glob("*.pdf"))
        if file_names is not None:
            pdf_files = [pdf_file for pdf_file in pdf_files if pdf_file.name in file_names]

        for pdf_file in pdf_files:
            reader = PdfReader(pdf_file)

            page_texts: list[str] = []

            for page in reader.pages:
                page_text = page.extract_text()

                page_texts.append(TextCleaner.clean(page_text or ""))

            clean_text = "\n\n".join(page_text for page_text in page_texts if page_text)

            document = Document(file_name=pdf_file.name, file_path=str(pdf_file), page_count=len(reader.pages), text=clean_text, page_texts=page_texts)
            documents.append(document)

        return documents
