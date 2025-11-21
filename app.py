import os
import streamlit as st
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter  # updated import
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document  # unified import for modular 0.1.17 stack
from chromadb.config import Settings
import datetime
import json
import httpx  # added for Gen AI Lab client
import requests  # added
import tiktoken  # added
from io import BytesIO

# NOTE: App built against pinned versions: langchain==0.1.17 and related split packages.
# Avoid using newer callback or Runnable features not present in this version.
# Ensure fpdf is installed for export (listed in requirements). If missing, install: pip install fpdf

# --- Config ---
DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
EXPORT_DIR = "exports"
TOKEN_FILE = os.path.join("token", os.listdir("token")[0]) if os.path.isdir("token") and os.listdir("token") else None
API_KEY_FILE = "api_key.txt"

os.makedirs(EXPORT_DIR, exist_ok=True)

BASE_URL = "https://genailab.tcs.in"  # Handbook-approved endpoint
ALLOWED_CHAT_MODELS = [
    "azure/genailab-maas-gpt-4o-mini",
    "azure/genailab-maas-gpt-4o",
    "azure_ai/genailab-maas-DeepSeek-V3-0324",
    "azure_ai/genailab-maas-DeepSeek-R1",
    "azure_ai/genailab-maas-Llama-3.3-70B-Instruct",
]
EMBEDDING_MODEL = "azure/genailab-maas-text-embedding-3-large"
TIKTOKEN_CACHE_DIR = os.path.join("token")
os.environ.setdefault("TIKTOKEN_CACHE_DIR", TIKTOKEN_CACHE_DIR)

def ensure_tiktoken_local():
    # Ensure encoding file exists locally to bypass SSL fetch
    enc_file = os.path.join(TIKTOKEN_CACHE_DIR, "cl100k_base.tiktoken")
    if os.path.exists(enc_file):
        return
    os.makedirs(TIKTOKEN_CACHE_DIR, exist_ok=True)
    try:
        # Attempt normal load (will download & cache if certificates OK)
        tiktoken.get_encoding("cl100k_base")
    except Exception:
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        try:
            resp = requests.get(url, timeout=15, verify=False)
            if resp.ok:
                with open(enc_file, "wb") as f:
                    f.write(resp.content)
        except Exception:
            pass  # If still failing, embeddings will fall back; chunk sizes small

ensure_tiktoken_local()

# --- Utils ---
def read_api_key() -> str:
    # expects line like: API-key = "sk-..."
    if not os.path.exists(API_KEY_FILE):
        return os.getenv("OPENAI_API_KEY", "")
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if "=" in content:
        return content.split("=")[-1].strip().strip('"')
    return content

API_KEY = read_api_key()
os.environ["OPENAI_API_KEY"] = API_KEY

def get_http_client():
    return httpx.Client(verify=False)

# Auto-load token (no user prompt)
APP_TOKEN = None
if TOKEN_FILE and os.path.exists(TOKEN_FILE):
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        APP_TOKEN = f.read().strip()

# --- Ingestion helper (quick) ---
def load_documents() -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(DATA_DIR):
        return docs
    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            path = os.path.join(root, name)
            if name.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                except Exception as e:
                    st.warning(f"Failed PDF {name}: {e}")
            elif name.lower().endswith(('.txt', '.md')):
                try:
                    loader = TextLoader(path, encoding='utf-8')
                    docs.extend(loader.load())
                except Exception as e:
                    st.warning(f"Failed text {name}: {e}")
    return docs

@st.cache_resource(show_spinner=False)
def build_vectorstore() -> Chroma:
    if not os.path.isdir(DATA_DIR):
        return None
    docs = load_documents()
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()  # updated
    vectorstore = Chroma(collection_name="reg_docs", persist_directory=VECTOR_DIR, embedding_function=embeddings,
                         client_settings=Settings(anonymized_telemetry=False))
    # Add only if empty (internal _collection usage stable in pinned version)
    try:
        if vectorstore._collection.count() == 0:
            vectorstore.add_documents(chunks)
    except Exception:
        vectorstore.add_documents(chunks)
    return vectorstore

@st.cache_resource(show_spinner=False)
def get_llm(selected_model: str):
    return ChatOpenAI(
        base_url=BASE_URL,
        model=selected_model,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        http_client=get_http_client(),
        temperature=0.1,
    )

def get_embeddings():
    return OpenAIEmbeddings(
        base_url=BASE_URL,
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        http_client=get_http_client(),
    )

# --- Retrieval + Answer ---

def answer_query(query: str, vectorstore: Chroma, selected_model: str) -> Dict:
    # Use direct similarity_search for compatibility with pinned versions
    docs = vectorstore.similarity_search(query, k=5) if vectorstore else []
    context = "\n\n".join([f"Source {i+1} ({d.metadata.get('source','unknown')}):\n{d.page_content}" for i, d in enumerate(docs)])
    system_prompt = (
        "You are a compliance assistant for capital markets. Answer the user question using ONLY the provided regulatory context. "
        "Cite sources by filename. If unsure, state uncertainty and suggest needed documents."
    )
    llm = get_llm(selected_model)
    messages = [
        ("system", system_prompt),
        ("user", f"Context:\n{context}\n\nQuestion: {query}\nProvide: concise answer + obligations summary + cited excerpts.")
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "sources": [d.metadata.get("source", "") for d in docs], "raw_docs": docs}

def answer_query_direct(query: str, docs: List[Document], selected_model: str) -> Dict:
    """Answer a query directly over a list of Documents (used for ad-hoc uploaded files)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    context = "\n\n".join([f"Chunk {i+1} ({d.metadata.get('source','uploaded')}):\n{d.page_content}" for i, d in enumerate(chunks)])
    system_prompt = (
        "You are a compliance assistant for capital markets. Answer the user question using ONLY the provided document context. "
        "Cite chunks by index. If unsure, state uncertainty."
    )
    llm = get_llm(selected_model)
    messages = [
        ("system", system_prompt),
        ("user", f"Context:\n{context}\n\nQuestion: {query}\nProvide: concise answer + obligations summary (if applicable) + cited excerpts."),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "sources": [d.metadata.get("source", "uploaded") for d in docs], "raw_docs": chunks}

# --- UI ---
st.set_page_config(page_title="Regulatory Compliance Query Agent", layout="wide")
st.title("Capital Markets Regulatory Compliance Query Agent")

if not API_KEY:
    st.error("Missing OpenAI API key. Add to api_key.txt or set OPENAI_API_KEY.")
    st.stop()

# Token status display only (no input required)
if APP_TOKEN:
    st.caption("Access token loaded from file.")
else:
    st.warning("Token file missing; proceeding without additional access gate.")

vectorstore = build_vectorstore()
if vectorstore is None:
    st.warning("No documents ingested yet. Add PDFs/TXT under data/ and reload.")
else:
    st.success("Vector store ready.")

if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts {q, a, sources, model, mode}

if "upload_chat" not in st.session_state:
    st.session_state.upload_chat = []  # separate history for uploaded docs

selected_model = st.selectbox("Model", ALLOWED_CHAT_MODELS, index=0, help="Models allowed per handbook")

# Tabs for two modes: backend corpus vs. uploaded document
mode = st.radio("Query mode", ["Backend documents", "Uploaded document"], horizontal=True)

if mode == "Backend documents":
    query = st.text_input("Ask a compliance question on ingested corpus", placeholder="e.g., What are liquidity coverage ratio requirements under Basel III?")
    if st.button("Submit", disabled=not vectorstore) and query.strip():
        with st.spinner("Retrieving & generating answer from backend corpus..."):
            result = answer_query(query, vectorstore, selected_model)
        st.session_state.chat.append({"q": query, "a": result["answer"], "sources": result["sources"], "model": selected_model, "mode": "backend"})

    for item in st.session_state.chat:
        st.markdown(f"**Mode:** Backend documents")
        st.markdown(f"**Model:** {item['model']}")
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(f"**A:** {item['a']}")
        st.markdown(f"Sources: {', '.join(item['sources'])}")
        st.markdown("---")

else:
    uploaded_file = st.file_uploader("Upload a regulation document (PDF or TXT)", type=["pdf", "txt"])
    upload_query = st.text_input("Ask a question about the uploaded document", placeholder="e.g., What are the key obligations in this document?")

    docs: List[Document] = []
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            # Save to a temporary in-memory buffer and use PyPDFLoader via file path workaround
            tmp_path = os.path.join("data", f"_tmp_upload_{uploaded_file.name}")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = uploaded_file.name
            # Clean-up file after load
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        else:
            # Assume text file
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            docs = [Document(page_content=content, metadata={"source": uploaded_file.name})]

    if st.button("Submit Question on Upload", disabled=uploaded_file is None or not upload_query.strip()):
        if not docs:
            st.error("Please upload a valid document first.")
        else:
            with st.spinner("Analyzing uploaded document..."):
                result = answer_query_direct(upload_query, docs, selected_model)
            st.session_state.upload_chat.append({"q": upload_query, "a": result["answer"], "sources": result["sources"], "model": selected_model, "mode": "upload"})

    for item in st.session_state.upload_chat:
        st.markdown(f"**Mode:** Uploaded document")
        st.markdown(f"**Model:** {item['model']}")
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(f"**A:** {item['a']}")
        st.markdown(f"Sources: {', '.join(item['sources'])}")
        st.markdown("---")

# Export
if st.session_state.chat:
    if st.button("Export Session"):
        from fpdf import FPDF
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Regulatory Compliance Q&A Session", ln=1)
            pdf.cell(0, 10, f"Generated: {datetime.datetime.utcnow().isoformat()} UTC", ln=1)
            pdf.ln(5)
            for i, item in enumerate(st.session_state.chat, start=1):
                pdf.multi_cell(0, 8, f"Q{i}: {item['q']}")
                pdf.multi_cell(0, 8, f"A{i}: {item['a']}")
                pdf.multi_cell(0, 8, f"Sources: {', '.join(item['sources'])}")
                pdf.ln(4)
            out_path = os.path.join(EXPORT_DIR, f"session_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf")
            pdf.output(out_path)
            st.success(f"Exported to {out_path}")
        except Exception as e:
            st.error(f"PDF export failed: {e}. Install fpdf with 'pip install fpdf'.")
            
st.caption("Hackathon prototype. Not legal advice.")
