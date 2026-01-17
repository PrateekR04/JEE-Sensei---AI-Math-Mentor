"""
User Profile
Student personalization and progress tracking.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict


class UserProfile:
    """
    Tracks user progress and personalizes experience.
    
    Stores per user:
    - Topics attempted
    - Accuracy per topic
    - Weak areas
    - Explanation preferences
    - Learning progress over time
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize user profile manager.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory_data"
            )
        
        self.storage_dir = storage_dir
        self.profiles_file = os.path.join(storage_dir, "user_profiles.json")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing profiles
        self.profiles: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        """Load profiles from disk."""
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r') as f:
                    self.profiles = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load user profiles: {e}")
                self.profiles = {}
    
    def _save(self):
        """Save profiles to disk."""
        try:
            with open(self.profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving user profiles: {e}")
    
    def _get_or_create(self, user_id: str) -> Dict:
        """Get existing profile or create new one."""
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "user_id": user_id,
                "created": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                
                # Topic tracking
                "topics": {},  # topic -> {attempts, correct, last_attempt}
                
                # Preferences
                "explanation_depth": "medium",  # brief, medium, detailed
                "preferred_method": None,  # substitution, elimination, etc.
                
                # History
                "total_problems": 0,
                "total_correct": 0,
                "streak": 0,
                "max_streak": 0,
                
                # Weak areas
                "weak_topics": [],
                "frequent_errors": [],
                
                # Session stats
                "sessions": [],
                
                # Recommendations
                "recommended_practice": []
            }
        
        return self.profiles[user_id]
    
    def record_attempt(self, user_id: str, topic: str, intent: str,
                       is_correct: bool, confidence: float, 
                       problem_id: str = None) -> Dict:
        """
        Record a problem attempt.
        
        Args:
            user_id: User identifier
            topic: Problem topic (algebra, calculus, etc.)
            intent: Problem intent (equation, derivative, etc.)
            is_correct: Whether the solution was correct
            confidence: Confidence score
            problem_id: Optional problem ID reference
            
        Returns:
            Updated user profile
        """
        profile = self._get_or_create(user_id)
        
        # Update last active
        profile["last_active"] = datetime.now().isoformat()
        
        # Update total counts
        profile["total_problems"] += 1
        if is_correct:
            profile["total_correct"] += 1
            profile["streak"] += 1
            profile["max_streak"] = max(profile["max_streak"], profile["streak"])
        else:
            profile["streak"] = 0
        
        # Update topic stats
        if topic not in profile["topics"]:
            profile["topics"][topic] = {
                "attempts": 0,
                "correct": 0,
                "intents": {},
                "last_attempt": None,
                "avg_confidence": 0.0
            }
        
        topic_stats = profile["topics"][topic]
        topic_stats["attempts"] += 1
        if is_correct:
            topic_stats["correct"] += 1
        topic_stats["last_attempt"] = datetime.now().isoformat()
        
        # Update running average confidence
        n = topic_stats["attempts"]
        old_avg = topic_stats["avg_confidence"]
        topic_stats["avg_confidence"] = (old_avg * (n - 1) + confidence) / n
        
        # Update intent stats
        if intent not in topic_stats["intents"]:
            topic_stats["intents"][intent] = {"attempts": 0, "correct": 0}
        topic_stats["intents"][intent]["attempts"] += 1
        if is_correct:
            topic_stats["intents"][intent]["correct"] += 1
        
        # Update weak topics
        profile["weak_topics"] = self._calculate_weak_topics(profile)
        
        # Update recommendations
        profile["recommended_practice"] = self._generate_recommendations(profile)
        
        self._save()
        return profile
    
    def _calculate_weak_topics(self, profile: Dict) -> List[str]:
        """Calculate weak topics based on accuracy."""
        weak = []
        
        for topic, stats in profile["topics"].items():
            if stats["attempts"] >= 3:  # Minimum attempts threshold
                accuracy = stats["correct"] / stats["attempts"]
                if accuracy < 0.6:  # Below 60% accuracy
                    weak.append({
                        "topic": topic,
                        "accuracy": accuracy,
                        "attempts": stats["attempts"]
                    })
        
        # Sort by accuracy (lowest first)
        weak.sort(key=lambda x: x["accuracy"])
        return [w["topic"] for w in weak[:5]]  # Top 5 weak topics
    
    def _generate_recommendations(self, profile: Dict) -> List[Dict]:
        """Generate practice recommendations."""
        recommendations = []
        
        # Recommend weak topics
        for topic in profile["weak_topics"][:3]:
            recommendations.append({
                "type": "practice",
                "topic": topic,
                "reason": "Low accuracy - needs practice"
            })
        
        # Recommend topics not attempted recently
        for topic, stats in profile["topics"].items():
            if stats.get("last_attempt"):
                last = datetime.fromisoformat(stats["last_attempt"])
                if datetime.now() - last > timedelta(days=7):
                    recommendations.append({
                        "type": "review",
                        "topic": topic,
                        "reason": "Not practiced recently"
                    })
        
        return recommendations[:5]
    
    def get_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile."""
        return self.profiles.get(user_id)
    
    def get_accuracy(self, user_id: str, topic: str = None) -> float:
        """Get accuracy for user, optionally filtered by topic."""
        profile = self.profiles.get(user_id)
        if not profile:
            return 0.0
        
        if topic and topic in profile["topics"]:
            stats = profile["topics"][topic]
            if stats["attempts"] > 0:
                return stats["correct"] / stats["attempts"]
            return 0.0
        
        if profile["total_problems"] > 0:
            return profile["total_correct"] / profile["total_problems"]
        return 0.0
    
    def get_explanation_depth(self, user_id: str) -> str:
        """Get preferred explanation depth for user."""
        profile = self.profiles.get(user_id)
        if profile:
            accuracy = self.get_accuracy(user_id)
            
            # Auto-adjust based on performance
            if accuracy < 0.5:
                return "detailed"  # More explanation for struggling users
            elif accuracy > 0.85:
                return "brief"  # Less for advanced users
            else:
                return profile.get("explanation_depth", "medium")
        
        return "medium"
    
    def set_preference(self, user_id: str, key: str, value: Any):
        """Set a user preference."""
        profile = self._get_or_create(user_id)
        profile[key] = value
        self._save()
    
    def get_summary(self, user_id: str) -> Dict:
        """Get a summary of user's progress."""
        profile = self.profiles.get(user_id)
        if not profile:
            return {"error": "User not found"}
        
        total = profile["total_problems"]
        correct = profile["total_correct"]
        
        # Topic breakdown
        topic_summary = {}
        for topic, stats in profile["topics"].items():
            if stats["attempts"] > 0:
                topic_summary[topic] = {
                    "attempts": stats["attempts"],
                    "accuracy": stats["correct"] / stats["attempts"],
                    "avg_confidence": stats["avg_confidence"]
                }
        
        return {
            "user_id": user_id,
            "total_problems": total,
            "overall_accuracy": correct / total if total > 0 else 0.0,
            "current_streak": profile["streak"],
            "max_streak": profile["max_streak"],
            "weak_topics": profile["weak_topics"],
            "topic_breakdown": topic_summary,
            "recommendations": profile["recommended_practice"],
            "explanation_depth": self.get_explanation_depth(user_id)
        }
    
    def list_users(self) -> List[str]:
        """List all user IDs."""
        return list(self.profiles.keys())


def main():
    """Test user profile."""
    profiles = UserProfile()
    
    # Record some attempts
    profiles.record_attempt("test_user", "algebra", "equation", True, 0.95)
    profiles.record_attempt("test_user", "algebra", "equation", True, 0.90)
    profiles.record_attempt("test_user", "algebra", "equation", False, 0.70)
    profiles.record_attempt("test_user", "calculus", "derivative", False, 0.60)
    profiles.record_attempt("test_user", "calculus", "derivative", False, 0.55)
    
    # Get summary
    summary = profiles.get_summary("test_user")
    print(f"User Summary:")
    print(f"  Total: {summary['total_problems']}")
    print(f"  Accuracy: {summary['overall_accuracy']:.0%}")
    print(f"  Streak: {summary['current_streak']}")
    print(f"  Weak Topics: {summary['weak_topics']}")
    print(f"  Explanation Depth: {summary['explanation_depth']}")


if __name__ == "__main__":
    main()
