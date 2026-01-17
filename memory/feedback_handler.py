"""
Feedback Handler
Processes user feedback and triggers learning updates.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional


class FeedbackHandler:
    """
    Handles user feedback (correct/incorrect) and corrections.
    
    Responsibilities:
    - Store feedback with timestamp
    - Link feedback to problem ID
    - Trigger learning updates (correction rules, patterns)
    - Track feedback statistics
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize feedback handler.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory_data"
            )
        
        self.storage_dir = storage_dir
        self.feedback_file = os.path.join(storage_dir, "feedback_log.json")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing feedback
        self.feedback_log: List[Dict] = []
        self._load()
        
        # Callbacks for learning updates
        self._on_correct_callbacks = []
        self._on_incorrect_callbacks = []
    
    def _load(self):
        """Load feedback log from disk."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    self.feedback_log = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load feedback log: {e}")
                self.feedback_log = []
    
    def _save(self):
        """Save feedback log to disk."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_log, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving feedback log: {e}")
    
    def register_on_correct(self, callback):
        """Register callback for correct feedback."""
        self._on_correct_callbacks.append(callback)
    
    def register_on_incorrect(self, callback):
        """Register callback for incorrect feedback."""
        self._on_incorrect_callbacks.append(callback)
    
    def store_feedback(self, problem_id: str, is_correct: bool,
                       correction: str = None, error_category: str = None,
                       problem_data: Dict = None, user_id: str = "default") -> str:
        """
        Store user feedback for a problem.
        
        Args:
            problem_id: ID of the problem
            is_correct: Whether the solution was correct
            correction: User's correction if incorrect
            error_category: Type of error (ocr, parsing, modeling, reasoning)
            problem_data: Full problem data for context
            user_id: User identifier
            
        Returns:
            Feedback ID
        """
        feedback_id = f"fb_{len(self.feedback_log)+1:06d}"
        
        feedback_record = {
            "feedback_id": feedback_id,
            "problem_id": problem_id,
            "is_correct": is_correct,
            "correction": correction,
            "error_category": error_category,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            
            # Include problem context for learning
            "problem_text": problem_data.get("problem_text", "") if problem_data else "",
            "original_answer": problem_data.get("answer", "") if problem_data else "",
            "topic": problem_data.get("topic", "") if problem_data else "",
            "intent": problem_data.get("intent", "") if problem_data else ""
        }
        
        self.feedback_log.append(feedback_record)
        self._save()
        
        # Trigger learning callbacks
        if is_correct:
            for callback in self._on_correct_callbacks:
                try:
                    callback(feedback_record)
                except Exception as e:
                    print(f"Error in correct callback: {e}")
        else:
            for callback in self._on_incorrect_callbacks:
                try:
                    callback(feedback_record)
                except Exception as e:
                    print(f"Error in incorrect callback: {e}")
        
        return feedback_id
    
    def get_feedback(self, problem_id: str) -> Optional[Dict]:
        """Get feedback for a specific problem."""
        for fb in reversed(self.feedback_log):  # Most recent first
            if fb.get("problem_id") == problem_id:
                return fb
        return None
    
    def get_corrections(self, error_category: str = None, 
                         limit: int = 100) -> List[Dict]:
        """
        Get corrections (incorrect feedback with corrections provided).
        
        Args:
            error_category: Filter by error type
            limit: Maximum number of corrections to return
        """
        corrections = []
        for fb in reversed(self.feedback_log):
            if not fb.get("is_correct") and fb.get("correction"):
                if error_category is None or fb.get("error_category") == error_category:
                    corrections.append(fb)
                    if len(corrections) >= limit:
                        break
        return corrections
    
    def get_stats(self, user_id: str = None) -> Dict:
        """Get feedback statistics."""
        if user_id:
            feedback = [f for f in self.feedback_log if f.get("user_id") == user_id]
        else:
            feedback = self.feedback_log
        
        total = len(feedback)
        if total == 0:
            return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0}
        
        correct = sum(1 for f in feedback if f.get("is_correct"))
        incorrect = total - correct
        
        # Error category breakdown
        error_categories = {}
        for f in feedback:
            if not f.get("is_correct"):
                cat = f.get("error_category", "unknown")
                error_categories[cat] = error_categories.get(cat, 0) + 1
        
        return {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": correct / total,
            "error_categories": error_categories
        }
    
    def get_recent(self, limit: int = 20) -> List[Dict]:
        """Get most recent feedback entries."""
        return list(reversed(self.feedback_log[-limit:]))
    
    def count(self) -> int:
        """Get total number of feedback entries."""
        return len(self.feedback_log)


def main():
    """Test feedback handler."""
    handler = FeedbackHandler()
    
    # Store correct feedback
    fb1 = handler.store_feedback(
        problem_id="test_001",
        is_correct=True,
        user_id="test_user"
    )
    print(f"Stored correct feedback: {fb1}")
    
    # Store incorrect feedback with correction
    fb2 = handler.store_feedback(
        problem_id="test_002",
        is_correct=False,
        correction="The correct answer is x = 3, not x = 2",
        error_category="reasoning",
        user_id="test_user"
    )
    print(f"Stored incorrect feedback: {fb2}")
    
    # Get stats
    stats = handler.get_stats()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
