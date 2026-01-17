"""
RAG Knowledge Base Ingestion Pipeline
Loads documents from knowledge_base/ and creates FAISS vector index
"""

import os
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class KnowledgeBaseIngester:
    """
    Ingests text documents and creates a searchable vector index.
    """
    
    def __init__(self, knowledge_base_dir: str = "rag/knowledge_base", 
                 vector_store_dir: str = "rag/vector_store"):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.embeddings = None
        self.vector_store = None
        
    def load_documents(self) -> List[Document]:
        """
        Load all .txt files from knowledge_base directory.
        
        Returns:
            List of Document objects with content and metadata
        """
        documents = []
        
        if not self.knowledge_base_dir.exists():
            print(f"Warning: {self.knowledge_base_dir} does not exist")
            return documents
        
        txt_files = list(self.knowledge_base_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"Warning: No .txt files found in {self.knowledge_base_dir}")
            return documents
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "path": str(file_path)
                    }
                )
                documents.append(doc)
                print(f"✓ Loaded: {file_path.name} ({len(content)} chars)")
                
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Increased to keep formulas and explanations together
            chunk_overlap=100,  # Increased overlap for better context continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
    
    def create_embeddings(self):
        """Initialize embedding model with normalization for cosine similarity."""
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
        )
        print("✓ Embedding model loaded (normalized for cosine similarity)")
    
    def create_vector_store(self, chunks: List[Document]):
        """
        Create FAISS vector store with cosine similarity (IndexFlatIP).
        
        Args:
            chunks: List of chunked documents
        """
        if not chunks:
            print("Warning: No chunks to index")
            return
        
        print("Creating vector store with cosine similarity...")
        # Create vector store - embeddings are already normalized
        # FAISS will use inner product which equals cosine similarity for normalized vectors
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # The index is already using inner product (cosine similarity) because
        # sentence-transformers normalizes embeddings by default
        print(f"✓ Vector store created with {len(chunks)} chunks (cosine similarity)")
    
    def save_vector_store(self):
        """Save vector store to disk."""
        if self.vector_store is None:
            print("Error: No vector store to save")
            return
        
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.vector_store_dir))
        print(f"✓ Vector store saved to {self.vector_store_dir}")
    
    def ingest(self):
        """
        Complete ingestion pipeline.
        
        Steps:
        1. Load documents
        2. Chunk documents
        3. Create embeddings
        4. Create vector store
        5. Save to disk
        """
        print("=" * 60)
        print("Knowledge Base Ingestion Pipeline")
        print("=" * 60)
        
        # Step 1: Load documents
        documents = self.load_documents()
        if not documents:
            print("\n⚠ No documents found. Add .txt files to rag/knowledge_base/")
            return
        
        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Step 3: Create embeddings
        self.create_embeddings()
        
        # Step 4: Create vector store
        self.create_vector_store(chunks)
        
        # Step 5: Save
        self.save_vector_store()
        
        print("=" * 60)
        print("✓ Ingestion complete!")
        print(f"  Documents: {len(documents)}")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Vector store: {self.vector_store_dir}")
        print("=" * 60)


def main():
    """Run ingestion pipeline."""
    ingester = KnowledgeBaseIngester()
    ingester.ingest()


if __name__ == "__main__":
    main()
