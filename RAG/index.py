from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_path = Path(__file__).parent / "Advance Prompt.pdf"

print("PDF path:", pdf_path)
print("PDF exists:", pdf_path.exists())

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

print("Pages loaded:", len(docs))
print(docs[2].page_content)