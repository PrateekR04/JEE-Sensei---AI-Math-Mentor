"""
Memory Store
Vector-based memory storage using ChromaDB for similarity search.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


from utils.embedding_loader import load_embedding_model

class MemoryStore:
    """
    Vector memory store for problem embeddings and similarity search.
    Uses ChromaDB for persistent storage.
    """
    
    def __init__(self, persist_dir: str = None):
        """
        Initialize memory store.
        
        Args:
            persist_dir: Directory for persistent storage
        """
        if persist_dir is None:
            persist_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory_data"
            )
        
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB
        if HAS_CHROMA:
            self.client = chromadb.PersistentClient(path=persist_dir)
            
            # Create collections for different memory types
            self.problems_collection = self.client.get_or_create_collection(
                name="solved_problems",
                metadata={"description": "Solved math problems with solutions"}
            )
            
            self.patterns_collection = self.client.get_or_create_collection(
                name="solution_patterns",
                metadata={"description": "Reusable solution patterns"}
            )
        else:
            # Fallback to in-memory JSON storage
            self.problems_file = os.path.join(persist_dir, "problems.json")
            self.patterns_file = os.path.join(persist_dir, "patterns.json")
            self._load_json_store()
        
        # Initialize embedding model using shared loader
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedder = load_embedding_model()
        else:
            self.embedder = None
        
        print(f"✓ Memory store initialized at {persist_dir}")
    
    def _load_json_store(self):
        """Load JSON-based fallback storage."""
        self.problems_store = []
        self.patterns_store = []
        
        if os.path.exists(self.problems_file):
            with open(self.problems_file, 'r') as f:
                self.problems_store = json.load(f)
        
        if os.path.exists(self.patterns_file):
            with open(self.patterns_file, 'r') as f:
                self.patterns_store = json.load(f)
    
    def _save_json_store(self):
        """Save JSON-based fallback storage."""
        with open(self.problems_file, 'w') as f:
            json.dump(self.problems_store, f, indent=2, default=str)
        
        with open(self.patterns_file, 'w') as f:
            json.dump(self.patterns_store, f, indent=2, default=str)
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self.embedder:
            return self.embedder.encode(text).tolist()
        else:
            # Simple hash-based pseudo-embedding for fallback
            import hashlib
            hash_obj = hashlib.sha256(text.encode())
            # Create a 384-dim vector from hash (matching MiniLM)
            hash_bytes = hash_obj.digest() * 12  # 32 * 12 = 384
            return [b / 255.0 for b in hash_bytes[:384]]
    
    def store_problem(self, problem_data: Dict[str, Any]) -> str:
        """
        Store a solved problem in memory.
        
        Args:
            problem_data: Dict containing problem details
            
        Returns:
            Problem ID
        """
        problem_text = problem_data.get("problem_text", "")
        problem_id = self._generate_id(problem_text + str(datetime.now()))
        
        # Create document
        document = {
            "id": problem_id,
            "problem_text": problem_text,
            "answer": problem_data.get("answer", ""),
            "topic": problem_data.get("topic", ""),
            "intent": problem_data.get("intent", ""),
            "equations": problem_data.get("equations", []),
            "solution_steps": problem_data.get("solution_steps", ""),
            "confidence": problem_data.get("confidence", 0.0),
            "is_correct": problem_data.get("is_correct", None),
            "timestamp": datetime.now().isoformat(),
            "user_id": problem_data.get("user_id", "default")
        }
        
        if HAS_CHROMA:
            embedding = self._get_embedding(problem_text)
            self.problems_collection.add(
                ids=[problem_id],
                embeddings=[embedding],
                documents=[problem_text],
                metadatas=[document]
            )
        else:
            document["embedding"] = self._get_embedding(problem_text)
            self.problems_store.append(document)
            self._save_json_store()
        
        return problem_id
    
    def find_similar(self, query: str, n_results: int = 5, 
                     threshold: float = 0.7) -> List[Dict]:
        """
        Find similar problems in memory.
        
        Args:
            query: Problem text to search for
            n_results: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar problems with scores
        """
        if HAS_CHROMA:
            embedding = self._get_embedding(query)
            results = self.problems_collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            similar = []
            if results["ids"] and results["ids"][0]:
                for i, id in enumerate(results["ids"][0]):
                    # ChromaDB returns L2 distance, convert to similarity
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= threshold:
                        similar.append({
                            "id": id,
                            "problem_text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "similarity": similarity
                        })
            
            return similar
        else:
            # Fallback: compute cosine similarity manually
            query_embedding = self._get_embedding(query)
            
            similar = []
            for problem in self.problems_store:
                prob_embedding = problem.get("embedding", [])
                if prob_embedding:
                    similarity = self._cosine_similarity(query_embedding, prob_embedding)
                    if similarity >= threshold:
                        similar.append({
                            "id": problem["id"],
                            "problem_text": problem["problem_text"],
                            "metadata": problem,
                            "similarity": similarity
                        })
            
            # Sort by similarity
            similar.sort(key=lambda x: x["similarity"], reverse=True)
            return similar[:n_results]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def store_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """
        Store a solution pattern for reuse.
        
        Args:
            pattern_data: Pattern details
            
        Returns:
            Pattern ID
        """
        pattern_text = pattern_data.get("pattern_description", "")
        pattern_id = self._generate_id(pattern_text)
        
        document = {
            "id": pattern_id,
            "pattern_description": pattern_text,
            "pattern_type": pattern_data.get("pattern_type", ""),
            "example_problems": pattern_data.get("example_problems", []),
            "solution_template": pattern_data.get("solution_template", ""),
            "success_count": pattern_data.get("success_count", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        if HAS_CHROMA:
            embedding = self._get_embedding(pattern_text)
            self.patterns_collection.upsert(
                ids=[pattern_id],
                embeddings=[embedding],
                documents=[pattern_text],
                metadatas=[document]
            )
        else:
            # Check if pattern exists
            existing = next((p for p in self.patterns_store if p["id"] == pattern_id), None)
            if existing:
                existing.update(document)
            else:
                document["embedding"] = self._get_embedding(pattern_text)
                self.patterns_store.append(document)
            self._save_json_store()
        
        return pattern_id
    
    def get_problem_count(self) -> int:
        """Get total number of stored problems."""
        if HAS_CHROMA:
            return self.problems_collection.count()
        else:
            return len(self.problems_store)
    
    def get_pattern_count(self) -> int:
        """Get total number of stored patterns."""
        if HAS_CHROMA:
            return self.patterns_collection.count()
        else:
            return len(self.patterns_store)
    
    def update_problem_feedback(self, problem_id: str, is_correct: bool, 
                                 correction: str = None) -> bool:
        """
        Update feedback for a stored problem.
        
        Args:
            problem_id: ID of the problem
            is_correct: Whether the solution was correct
            correction: User's correction if incorrect
            
        Returns:
            True if updated successfully
        """
        if HAS_CHROMA:
            try:
                result = self.problems_collection.get(ids=[problem_id])
                if result["ids"]:
                    metadata = result["metadatas"][0]
                    metadata["is_correct"] = is_correct
                    metadata["feedback_timestamp"] = datetime.now().isoformat()
                    if correction:
                        metadata["user_correction"] = correction
                    
                    self.problems_collection.update(
                        ids=[problem_id],
                        metadatas=[metadata]
                    )
                    return True
            except Exception as e:
                print(f"Error updating feedback: {e}")
                return False
        else:
            for problem in self.problems_store:
                if problem["id"] == problem_id:
                    problem["is_correct"] = is_correct
                    problem["feedback_timestamp"] = datetime.now().isoformat()
                    if correction:
                        problem["user_correction"] = correction
                    self._save_json_store()
                    return True
        
        return False


def main():
    """Test memory store."""
    store = MemoryStore()
    
    # Test storing a problem
    problem_id = store.store_problem({
        "problem_text": "Solve 2x + 3 = 7",
        "answer": "x = 2",
        "topic": "algebra",
        "confidence": 0.95
    })
    print(f"Stored problem: {problem_id}")
    
    # Test finding similar
    similar = store.find_similar("Solve 3x + 5 = 11", n_results=3)
    print(f"Found {len(similar)} similar problems")
    for s in similar:
        print(f"  - {s['problem_text'][:50]}... (similarity: {s['similarity']:.2f})")
    
    print(f"\nTotal problems: {store.get_problem_count()}")


if __name__ == "__main__":
    main()
