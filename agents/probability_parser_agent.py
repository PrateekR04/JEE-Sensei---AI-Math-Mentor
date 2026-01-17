"""
Probability Parser Agent
Parses probability problems from natural language
"""

import os
import re
import sys
from typing import Dict, Any, Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ProbabilityParserAgent:
    """
    Parses probability problems from natural language.
    Supports: basic probability, conditional, Bayes, combinations, permutations
    """
    
    def __init__(self):
        pass
    
    def _detect_probability_type(self, text: str) -> str:
        """
        Detect the type of probability problem.
        """
        text = text.lower()
        
        # Bayes detection
        if any(kw in text for kw in ['bayes', 'given that', 'posterior', 'prior']):
            return "bayes"
        
        # Conditional probability
        if any(kw in text for kw in ['given', 'conditional', 'if we know', 'knowing that', '|']):
            return "conditional"
        
        # Combinations
        if any(kw in text for kw in ['combination', 'choose', 'select', 'ncr', 'c(n,r)']):
            return "combinations"
        
        # Permutations
        if any(kw in text for kw in ['permutation', 'arrange', 'order matters', 'npr', 'p(n,r)']):
            return "permutations"
        
        # Binomial
        if any(kw in text for kw in ['exactly', 'out of', 'trials', 'success', 'flips', 'tosses', 'rolls']):
            return "binomial"
        
        # Basic probability
        return "basic"
    
    def _extract_numbers(self, text: str) -> List[int]:
        """
        Extract all numbers from text.
        """
        numbers = re.findall(r'\b(\d+)\b', text)
        return [int(n) for n in numbers]
    
    def _extract_coin_flip_params(self, text: str) -> Dict[str, Any]:
        """
        Extract parameters for coin flip problems.
        """
        text_lower = text.lower()
        numbers = self._extract_numbers(text)
        
        # Look for patterns like "2 heads in 3 flips"
        match = re.search(r'(\d+)\s*heads?\s*(?:in|out of)?\s*(\d+)\s*(?:flips?|tosses?|trials?)', text_lower)
        if match:
            return {
                "favorable": int(match.group(1)),
                "total_trials": int(match.group(2)),
                "event": "heads",
                "probability_per_trial": 0.5
            }
        
        # Look for "at least X heads"
        match = re.search(r'at least\s+(\d+)\s*heads?\s*(?:in|out of)?\s*(\d+)', text_lower)
        if match:
            return {
                "at_least": int(match.group(1)),
                "total_trials": int(match.group(2)),
                "event": "heads",
                "probability_per_trial": 0.5
            }
        
        return {"numbers": numbers}
    
    def _extract_dice_params(self, text: str) -> Dict[str, Any]:
        """
        Extract parameters for dice problems.
        """
        text_lower = text.lower()
        numbers = self._extract_numbers(text)
        
        # Look for sum patterns
        match = re.search(r'sum\s*(?:of|is|equals?)?\s*(\d+)', text_lower)
        if match:
            target_sum = int(match.group(1))
            # Count dice
            dice_count = 2  # Default
            if 'two dice' in text_lower or '2 dice' in text_lower:
                dice_count = 2
            elif 'three dice' in text_lower or '3 dice' in text_lower:
                dice_count = 3
            
            return {
                "dice_count": dice_count,
                "target_sum": target_sum,
                "sides": 6
            }
        
        return {"numbers": numbers}
    
    def _extract_card_params(self, text: str) -> Dict[str, Any]:
        """
        Extract parameters for card problems.
        """
        text_lower = text.lower()
        
        result = {"deck_size": 52}
        
        # Detect card types
        if 'ace' in text_lower:
            result["event"] = "ace"
            result["favorable"] = 4
        elif 'king' in text_lower:
            result["event"] = "king"
            result["favorable"] = 4
        elif 'queen' in text_lower:
            result["event"] = "queen"
            result["favorable"] = 4
        elif 'heart' in text_lower:
            result["event"] = "heart"
            result["favorable"] = 13
        elif 'spade' in text_lower:
            result["event"] = "spade"
            result["favorable"] = 13
        elif 'red' in text_lower:
            result["event"] = "red card"
            result["favorable"] = 26
        elif 'black' in text_lower:
            result["event"] = "black card"
            result["favorable"] = 26
        elif 'face card' in text_lower:
            result["event"] = "face card"
            result["favorable"] = 12
        
        return result
    
    def _extract_basic_probability(self, text: str) -> Dict[str, Any]:
        """
        Extract basic probability parameters.
        """
        text_lower = text.lower()
        numbers = self._extract_numbers(text)
        
        # Look for "X favorable out of Y total"
        match = re.search(r'(\d+)\s*(?:favorable|winning|success)\s*(?:out of|from|in)\s*(\d+)', text_lower)
        if match:
            return {
                "favorable": int(match.group(1)),
                "total": int(match.group(2))
            }
        
        # Look for "X out of Y"
        match = re.search(r'(\d+)\s*(?:out of|from|in)\s*(\d+)', text_lower)
        if match:
            return {
                "favorable": int(match.group(1)),
                "total": int(match.group(2))
            }
        
        return {"numbers": numbers}
    
    def parse(self, problem_text: str) -> Dict[str, Any]:
        """
        Parse probability problem from natural language.
        
        Args:
            problem_text: User's problem text
            
        Returns:
            Dict with type, extracted parameters, and any errors
        """
        text_lower = problem_text.lower()
        prob_type = self._detect_probability_type(problem_text)
        
        result = {
            "type": prob_type,
            "problem_text": problem_text,
            "error": None,
            "needs_clarification": False
        }
        
        # Detect specific scenarios
        if 'coin' in text_lower or 'flip' in text_lower or 'toss' in text_lower:
            result["scenario"] = "coin"
            result["params"] = self._extract_coin_flip_params(problem_text)
        
        elif 'dice' in text_lower or 'die' in text_lower:
            result["scenario"] = "dice"
            result["params"] = self._extract_dice_params(problem_text)
        
        elif 'card' in text_lower or 'deck' in text_lower:
            result["scenario"] = "cards"
            result["params"] = self._extract_card_params(problem_text)
        
        else:
            result["scenario"] = "general"
            result["params"] = self._extract_basic_probability(problem_text)
        
        # Add description
        result["description"] = self._generate_description(result)
        
        return result
    
    def _generate_description(self, parsed: Dict) -> str:
        """
        Generate a description of the probability problem.
        """
        scenario = parsed.get("scenario", "general")
        params = parsed.get("params", {})
        
        if scenario == "coin":
            if "favorable" in params and "total_trials" in params:
                return f"Getting {params['favorable']} heads in {params['total_trials']} coin flips"
            elif "at_least" in params:
                return f"Getting at least {params['at_least']} heads in {params['total_trials']} coin flips"
        
        elif scenario == "dice":
            if "target_sum" in params:
                return f"Rolling a sum of {params['target_sum']} with {params.get('dice_count', 2)} dice"
        
        elif scenario == "cards":
            if "event" in params:
                return f"Drawing a {params['event']} from a deck of {params.get('deck_size', 52)} cards"
        
        return parsed.get("problem_text", "")[:50]


def main():
    """Test probability parser."""
    parser = ProbabilityParserAgent()
    
    test_cases = [
        "What is the probability of getting 2 heads in 3 coin flips?",
        "Probability of rolling a sum of 7 with two dice",
        "What is the probability of drawing an ace from a standard deck?",
        "If 3 items are defective out of 20, what is the probability of picking a defective item?",
        "Probability of getting at least 2 heads in 4 tosses"
    ]
    
    print("Probability Parser Test Results:")
    print("=" * 70)
    
    for case in test_cases:
        print(f"\nInput: {case}")
        result = parser.parse(case)
        print(f"  Type: {result['type']}")
        print(f"  Scenario: {result.get('scenario', 'N/A')}")
        print(f"  Params: {result.get('params', {})}")
        print(f"  Description: {result.get('description', 'N/A')}")


if __name__ == "__main__":
    main()
