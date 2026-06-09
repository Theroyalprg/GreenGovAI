"""
rag.py — RAG Pipeline for GreenGov AI
Handles PDF ingestion, chunking, FAISS vector store, and retrieval.
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

# Updated imports for newer LangChain versions
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DATA_DIR = Path("data")
CACHE_DIR = Path(".cache")
VECTORSTORE_PATH = CACHE_DIR / "faiss_index"
HASH_CACHE_PATH = CACHE_DIR / "pdf_hash.pkl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _compute_pdf_hash(pdf_paths: List[Path]) -> str:
    """Compute a combined MD5 hash of all PDF files to detect changes."""
    hasher = hashlib.md5()
    for path in sorted(pdf_paths):
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def _load_cached_hash() -> Optional[str]:
    """Load the previously stored PDF hash (if any)."""
    if HASH_CACHE_PATH.exists():
        try:
            with open(HASH_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_hash(hash_val: str) -> None:
    """Persist the current PDF hash."""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(HASH_CACHE_PATH, "wb") as f:
        pickle.dump(hash_val, f)


# ─────────────────────────────────────────────
# PDF Loading & Chunking
# ─────────────────────────────────────────────

def load_pdf_documents() -> List[Document]:
    """
    Load all PDFs from the /data directory.
    Returns a flat list of LangChain Document objects.
    """
    pdf_paths = list(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        return []

    documents: List[Document] = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            # Tag each page with its source scheme name
            for page in pages:
                page.metadata["scheme_name"] = pdf_path.stem
                if "source" not in page.metadata:
                    page.metadata["source"] = pdf_path.name
            documents.extend(pages)
            print(f"[RAG] Loaded {pdf_path.name} ({len(pages)} pages)")
        except Exception as e:
            print(f"[RAG] Warning: Could not load {pdf_path.name}: {e}")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[RAG] Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


# ─────────────────────────────────────────────
# Vector Store Management
# ─────────────────────────────────────────────

def build_vectorstore(chunks: List[Document], api_key: str) -> FAISS:
    """
    Embed chunks and store in a FAISS vector database.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """Persist FAISS index to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_PATH))


def load_vectorstore(api_key: str) -> Optional[FAISS]:
    """Load FAISS index from disk if it exists."""
    if not VECTORSTORE_PATH.exists():
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore
    except Exception as e:
        print(f"[RAG] Could not load cached vectorstore: {e}")
        return None


# ─────────────────────────────────────────────
# Main Initialization Entry Point
# ─────────────────────────────────────────────

def initialize_rag(api_key: str) -> Tuple[Optional[FAISS], List[str], str]:
    """
    Full RAG initialization:
    1. Detect PDFs in /data
    2. Check if FAISS cache is still valid
    3. Rebuild if needed
    4. Return (vectorstore, scheme_names, status_message)
    """
    if not api_key:
        return None, [], "no_api_key"
    
    pdf_paths = list(DATA_DIR.glob("*.pdf"))
    scheme_names = [p.stem for p in pdf_paths]

    if not pdf_paths:
        return None, [], "no_pdfs"

    current_hash = _compute_pdf_hash(pdf_paths)
    cached_hash = _load_cached_hash()

    # Use cached FAISS if PDFs haven't changed
    if current_hash == cached_hash:
        vectorstore = load_vectorstore(api_key)
        if vectorstore:
            return vectorstore, scheme_names, "loaded_cache"

    # Rebuild from scratch
    print("[RAG] Building new vector store...")
    documents = load_pdf_documents()
    if not documents:
        return None, scheme_names, "no_documents_loaded"
    
    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks, api_key)
    save_vectorstore(vectorstore)
    _save_hash(current_hash)

    return vectorstore, scheme_names, f"built_new ({len(chunks)} chunks)"


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    vectorstore: FAISS,
    k: int = TOP_K_RETRIEVAL,
) -> List[Document]:
    """
    Retrieve the top-k most relevant document chunks for a given query.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever.invoke(query)


def format_context(docs: List[Document]) -> Tuple[str, List[str]]:
    """
    Format retrieved documents into a context string and source list.
    Returns (context_text, sources_list).
    """
    if not docs:
        return "No relevant documents found.", []
    
    context_parts = []
    sources = []

    for i, doc in enumerate(docs, 1):
        scheme = doc.metadata.get("scheme_name", "Unknown Scheme")
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", f"{scheme}.pdf")
        
        context_parts.append(
            f"[Source {i} — {scheme}, Page {page}]\n{doc.page_content.strip()[:1000]}"
        )
        source_label = f"{scheme} ({source}, page {page})"
        if source_label not in sources:
            sources.append(source_label)

    return "\n\n---\n\n".join(context_parts), sources
