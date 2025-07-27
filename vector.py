from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob

# === Step 1: Set PDF Folder and Load All PDFs ===
pdf_folder = "./pdfs"  # Folder where your 5 PDFs are stored
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

all_documents = []

for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    # Split each PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(raw_docs)

    # Optionally add metadata (like source file name)
    for doc in split_docs:
        doc.metadata["source"] = os.path.basename(pdf_path)

    all_documents.extend(split_docs)

# === Step 2: Set up Embeddings ===
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# === Step 3: Set up Vector Store ===
db_location = "./chroma_langchain_pdf_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="pdf_documents",
    persist_directory=db_location,
    embedding_function=embeddings
)

# === Step 4: Add All Documents ===
if add_documents:
    ids = [str(i) for i in range(len(all_documents))]
    vector_store.add_documents(documents=all_documents, ids=ids)

# === Step 5: Create Retriever ===
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# === Step 6: Query ===
query = "What types of treatments or coverage does this insurance offer?"
results = retriever.invoke(query)

if results:
    print("Top result:")
    print(results[0].page_content)
else:
    print("No results found for the query.")

