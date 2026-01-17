"""
RAG Retriever
Semantic search over indexed knowledge base
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


from utils.embedding_loader import get_local_model_path

class KnowledgeRetriever:
    """
    Retrieves relevant knowledge chunks from vector store.
    """
    
    def __init__(self, vector_store_dir: str = "rag/vector_store"):
        self.vector_store_dir = Path(vector_store_dir)
        self.embeddings = None
        self.vector_store = None
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load FAISS vector store from disk with normalized embeddings."""
        if not self.vector_store_dir.exists():
            print(f"Warning: Vector store not found at {self.vector_store_dir}")
            print("Run 'python rag/ingest.py' to create the index")
            return
        
        try:
            # Get local model path (automatically downloads if needed)
            local_model_path = get_local_model_path()
            
            # Initialize embeddings with normalization (must match ingestion)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=local_model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
            )
            
            # Load vector store
            self.vector_store = FAISS.load_local(
                str(self.vector_store_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✓ Vector store loaded successfully (cosine similarity mode)")
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.vector_store = None
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            
        Returns:
            List of dicts with 'content' and 'source'
        """
        if self.vector_store is None:
            return []
        
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Format results
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown")
                })
            
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def retrieve_with_threshold(self, query: str, k: int = 3, 
                                threshold: float = 0.35) -> Dict[str, Any]:
        """
        Retrieve chunks with similarity threshold for strict RAG mode.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            threshold: Minimum similarity score (0.0 to 1.0)
                      0.35 = minimum acceptable match
                      0.5 = good match
                      0.8+ = excellent match
            
        Returns:
            Dict with 'results', 'has_sufficient_context', and 'best_score'
        """
        if self.vector_store is None:
            return {
                "results": [],
                "has_sufficient_context": False,
                "best_score": 0.0,
                "reason": "Vector store not loaded"
            }
        
        try:
            # Perform similarity search with scores
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            if not docs_and_scores:
                return {
                    "results": [],
                    "has_sufficient_context": False,
                    "best_score": 0.0,
                    "reason": "No documents found"
                }
            
            # FAISS with normalized embeddings uses L2 distance
            # L2 distance on normalized vectors: distance = sqrt(2 * (1 - cosine_similarity))
            # Therefore: cosine_similarity = 1 - (distance^2 / 2)
            # For small distances: similarity ≈ 1 - distance/2
            
            filtered_results = []
            similarities = []
            
            for doc, l2_distance in docs_and_scores:
                # Convert L2 distance to approximate cosine similarity
                # L2 distance range: 0 (perfect match) to 2 (opposite)
                # Convert to similarity: 0 → 1.0, 2 → 0.0
                similarity = max(0.0, 1.0 - (l2_distance / 2.0))
                similarities.append(similarity)
                
                if similarity >= threshold:
                    filtered_results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "score": float(similarity),
                        "l2_distance": float(l2_distance)
                    })
            
            # Get best score (highest similarity)
            best_score = max(similarities) if similarities else 0.0
            
            # Determine if context is sufficient
            has_sufficient = len(filtered_results) > 0 and best_score >= threshold
            
            reason = ""
            if not has_sufficient:
                if not filtered_results:
                    reason = f"No results above threshold {threshold} (best similarity: {best_score:.3f})"
                else:
                    reason = f"Best match similarity {best_score:.3f} below threshold {threshold}"
            
            return {
                "results": filtered_results,
                "has_sufficient_context": has_sufficient,
                "best_score": float(best_score),
                "reason": reason
            }
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return {
                "results": [],
                "has_sufficient_context": False,
                "best_score": float('inf'),
                "reason": f"Error: {str(e)}"
            }
    
    def retrieve_with_scores(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve chunks with similarity scores.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            
        Returns:
            List of dicts with 'content', 'source', and 'score'
        """
        if self.vector_store is None:
            return []
        
        try:
            # Perform similarity search with scores
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            results = []
            for doc, score in docs_and_scores:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": float(score)
                })
            
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if vector store is loaded and ready."""
        return self.vector_store is not None
    
    def has_sufficient_context(self, query: str, threshold: float = 0.3) -> bool:
        """
        Quick check if retrieval can find sufficient context for a query.
        
        Args:
            query: Search query
            threshold: Minimum similarity threshold
            
        Returns:
            True if sufficient context found, False otherwise
        """
        result = self.retrieve_with_threshold(query, k=3, threshold=threshold)
        return result["has_sufficient_context"]


def main():
    """Test retrieval."""
    retriever = KnowledgeRetriever()
    
    if not retriever.is_available():
        print("Vector store not available. Run 'python rag/ingest.py' first.")
        return
    
    # Test query
    query = "How do I solve a linear equation?"
    print(f"\nQuery: {query}")
    print("=" * 60)
    
    results = retriever.retrieve(query, k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Source: {result['source']}")
        print(f"Content: {result['content'][:200]}...")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
