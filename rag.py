"""
RAG Pipeline Module for GreenGov AI
Handles PDF loading, chunking, embedding generation, and FAISS vector storage
"""

import os
from typing import List, Tuple
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import google.generativeai as genai

class RAGPipeline:
    """Manages document ingestion, embedding, and retrieval for sustainability schemes"""
    
    def __init__(self, api_key: str, data_path: str = "./data"):
        self.api_key = api_key
        self.data_path = Path(data_path)
        self.embeddings = None
        self.vector_store = None
        self.documents = []
        
        # Initialize embeddings
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
    
    def load_pdfs(self) -> List[Document]:
        """Load all PDF documents from the data directory"""
        documents = []
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        pdf_files = list(self.data_path.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_path}")
        
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                
                # Add metadata to each document
                for doc in docs:
                    doc.metadata["source"] = pdf_path.name
                    doc.metadata["scheme_name"] = pdf_path.stem.replace("_", " ").title()
                
                documents.extend(docs)
                print(f"✓ Loaded: {pdf_path.name} ({len(docs)} pages)")
                
            except Exception as e:
                print(f"✗ Error loading {pdf_path.name}: {str(e)}")
        
        self.documents = documents
        return documents
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Split documents into smaller chunks for better retrieval"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vector store from document chunks"""
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Check API key.")
        
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print(f"✓ Created FAISS vector store with {len(chunks)} vectors")
        
        return self.vector_store
    
    def load_or_create_index(self, index_path: str = None) -> FAISS:
        """Load existing index or create new one from PDFs"""
        if index_path and Path(index_path).exists():
            try:
                self.vector_store = FAISS.load_local(
                    index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✓ Loaded existing index from {index_path}")
                return self.vector_store
            except Exception as e:
                print(f"Could not load index: {e}")
        
        # Create new index
        documents = self.load_pdfs()
        chunks = self.chunk_documents(documents)
        self.create_vector_store(chunks)
        
        # Save index if path provided
        if index_path:
            self.vector_store.save_local(index_path)
            print(f"✓ Saved index to {index_path}")
        
        return self.vector_store
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve top-k relevant chunks for a given query"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or create index first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        print(f"✓ Retrieved {len(results)} relevant chunks for: {query[:50]}...")
        
        return results
    
    def get_scheme_names(self) -> List[str]:
        """Get list of all scheme names from loaded documents"""
        if not self.documents:
            return []
        
        schemes = set()
        for doc in self.documents:
            if "scheme_name" in doc.metadata:
                schemes.add(doc.metadata["scheme_name"])
        
        return sorted(list(schemes))
