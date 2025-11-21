import os
from langchain_text_splitters import RecursiveCharacterTextSplitter  # updated import for 0.1.17 modular packages
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"

API_KEY_FILE = "api_key.txt"

# Using versions pinned to langchain==0.1.17 and companion split packages for hackathon stability.
# Avoid newer APIs not present in that version range.

def read_api_key():
    if not os.path.exists(API_KEY_FILE):
        return os.getenv("OPENAI_API_KEY", "")
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if "=" in content:
        return content.split("=")[-1].strip().strip('"')
    return content

os.environ["OPENAI_API_KEY"] = read_api_key()


def load_documents():
    docs = []
    if not os.path.isdir(DATA_DIR):
        print("Create a data/ folder and add PDF or TXT files.")
        return docs
    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            path = os.path.join(root, name)
            if name.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Failed PDF {name}: {e}")
            elif name.lower().endswith((".txt", ".md")):
                try:
                    loader = TextLoader(path, encoding="utf-8")
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Failed text {name}: {e}")
    return docs


def main():
    docs = load_documents()
    if not docs:
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(collection_name="reg_docs", persist_directory=VECTOR_DIR, embedding_function=embeddings, client_settings=Settings(anonymized_telemetry=False))
    # Only add if collection is empty (compat guard for 0.1.17 internal API changes)
    try:
        current_count = vectorstore._collection.count()  # internal but stable in pinned version
    except Exception:
        current_count = 0
    if current_count == 0:
        vectorstore.add_documents(chunks)
    else:
        print("Vector store already populated; skipping re-add.")
    print(f"Ingested {len(chunks)} chunks (may have been already present). Persisted at {VECTOR_DIR}/")


if __name__ == "__main__":
    main()
