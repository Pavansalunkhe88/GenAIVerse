from dotenv import load_dotenv
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore


load_dotenv()

# create gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768
)

# connect to Qdrant vector store
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="advance_prompt",
    url="http://localhost:6333"
)

# user question
query = input("\nAsk a question: ")

# similarity search in Qdrant
results = vector_store.similarity_search(
    query,
    k=3
)

#display results
print("\n========== RETRIEVED CHUNKS ==========\n")

for i, result in enumerate(results, start=1):

    print(f"--- Chunk {i} ---")

    print(result.page_content)

    print("\nMetadata:")
    print(result.metadata)

    print("\n" + "=" * 50)