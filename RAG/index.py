from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = Path(__file__).parent / "Advance Prompt.pdf"

print("PDF path:", pdf_path)
print("PDF exists:", pdf_path.exists())


# Load the PDF file
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

print("Pages loaded:", len(docs))
print(docs[2].page_content)

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

texts = text_splitter.split_documents(docs)

#vector embedding
