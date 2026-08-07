from pathlib import Path
from collections.abc import Callable

from pypdf import PdfReader

from models import Document
from services.text_cleaner import TextCleaner


class DocumentLoader:

    def __init__(self, documents_path: str = "data/documents"):
        self.documents_path = Path(documents_path)


    def get_document_names(self) -> set[str]:
        return {pdf_file.name for pdf_file in self.documents_path.glob("*.pdf")}

    def load_documents(
        self,
        file_names: set[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Document]:
        documents: list[Document] = []

        pdf_files = sorted(self.documents_path.glob("*.pdf"))
        if file_names is not None:
            pdf_files = [pdf_file for pdf_file in pdf_files if pdf_file.name in file_names]

        total_files = len(pdf_files)
        for index, pdf_file in enumerate(pdf_files, start=1):
            if progress_callback:
                progress_callback(index - 1, total_files, f"Reading {pdf_file.name}")
            reader = PdfReader(pdf_file)

            page_texts: list[str] = []

            for page in reader.pages:
                page_text = page.extract_text()

                page_texts.append(TextCleaner.clean(page_text or ""))

            clean_text = "\n\n".join(page_text for page_text in page_texts if page_text)

            document = Document(file_name=pdf_file.name, file_path=str(pdf_file), page_count=len(reader.pages), text=clean_text, page_texts=page_texts)
            documents.append(document)
            if progress_callback:
                progress_callback(index, total_files, f"Read {pdf_file.name}")

        return documents
