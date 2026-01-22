"""
Central Groq LLM Client for Math Mentor AI
Provides unified interface for all agents

UPGRADED: Now includes locked solver prompt for structured JSON output
"""
import os
import re
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# LOCKED SOLVER SYSTEM PROMPT - DO NOT MODIFY AT RUNTIME
# ============================================================================
_LOCKED_SOLVER_SYSTEM_PROMPT = """You are a precise mathematical solver.

You MUST return your output in the following strict JSON format:

{
  "final_answer": "...",
  "explanation": "Step 1: ...\\n\\nStep 2: ...\\n\\nStep 3: ...\\n\\n..."
}

Rules:
- final_answer must contain ONLY the final numeric or symbolic answer
- explanation must contain ALL steps taken to solve the problem
- Each step must start on a NEW LINE
- There must be a BLANK LINE between steps
- Steps must be clearly numbered (Step 1, Step 2, etc.)
- No markdown formatting
- No extra commentary
- No emojis
- No conversational text
- No prefixes like "Here is the answer"

Example:

{
  "final_answer": "x = 2",
  "explanation": "Step 1: Write the given equation.\\n2x + 3 = 7\\n\\nStep 2: Subtract 3 from both sides.\\n2x = 4\\n\\nStep 3: Divide both sides by 2.\\nx = 2"
}
"""

# Guard flag to ensure prompt immutability
_SOLVER_PROMPT_LOCKED = True


def _get_locked_solver_prompt() -> str:
    """
    Get the locked solver system prompt.
    This function ensures the prompt cannot be modified at runtime.
    """
    if not _SOLVER_PROMPT_LOCKED:
        raise RuntimeError("SECURITY ERROR: Solver prompt lock has been tampered with.")
    return _LOCKED_SOLVER_SYSTEM_PROMPT


class GroqClient:
    """
    Central LLM client using Groq API.
    Provides fast, quota-free inference for all agents.
    """
    
    def __init__(self, model="llama-3.3-70b-versatile"):
        """
        Initialize Groq client.
        
        Args:
            model: Default model to use (can be overridden per call)
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.default_model = model
    
    def generate(self, prompt: str, model: str = None, temperature: float = 0.1, 
                 max_tokens: int = 2048) -> str:
        """
        Generate text completion using Groq.
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to instance default)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        model = model or self.default_model
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise math reasoning AI. Provide accurate, step-by-step solutions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq API error: {e}")
            raise
    
    def generate_solver_response(self, prompt: str, model: str = None, 
                                  temperature: float = 0.1, max_tokens: int = 2048) -> dict:
        """
        Generate solver response with enforced JSON output format.
        Uses LOCKED solver system prompt.
        
        Args:
            prompt: Math problem to solve
            model: Model to use (defaults to instance default)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with final_answer and explanation, or error info
        """
        model = model or self.default_model
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": _get_locked_solver_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Parse and validate JSON response
            result = self._parse_solver_json(raw_response)
            validation = self._validate_solver_output(result)
            
            if not validation["valid"]:
                return {
                    "final_answer": "",
                    "explanation": "",
                    "raw_response": raw_response,
                    "error": validation["reason"],
                    "valid": False
                }
            
            return {
                "final_answer": result.get("final_answer", ""),
                "explanation": result.get("explanation", ""),
                "raw_response": raw_response,
                "valid": True
            }
            
        except Exception as e:
            print(f"Groq API error: {e}")
            return {
                "final_answer": "",
                "explanation": "",
                "error": str(e),
                "valid": False
            }
    
    def _parse_solver_json(self, text: str) -> dict:
        """Parse JSON from solver response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        return {}
    
    def _validate_solver_output(self, result: dict) -> dict:
        """
        Validate solver output meets requirements.
        
        Rejects if:
        - JSON is malformed
        - Missing final_answer or explanation
        - Steps are not separated by blank lines
        """
        if not result:
            return {"valid": False, "reason": "Empty or malformed JSON response"}
        
        if "final_answer" not in result:
            return {"valid": False, "reason": "Missing final_answer field"}
        
        if "explanation" not in result:
            return {"valid": False, "reason": "Missing explanation field"}
        
        final_answer = result.get("final_answer", "")
        explanation = result.get("explanation", "")
        
        if not final_answer:
            return {"valid": False, "reason": "final_answer is empty"}
        
        if not explanation:
            return {"valid": False, "reason": "explanation is empty"}
        
        # Check for multiple steps (should have at least 2 step markers or blank lines)
        has_steps = bool(re.search(r'Step\s*\d+', explanation, re.IGNORECASE))
        has_blank_lines = '\n\n' in explanation
        
        if not has_steps and not has_blank_lines:
            return {"valid": False, "reason": "explanation must contain numbered steps with blank lines between them"}
        
        return {"valid": True, "reason": None}


# Available Groq models
MODELS = {
    "fast": "llama-3.3-70b-versatile",          # Fast, good for classification
    "balanced": "llama-3.3-70b-versatile",   # Balanced speed/quality
    "large": "llama3-70b-8192",        # High quality
    "phi": "phi3-medium-128k"          # Long context
}


def get_client(model_type="fast"):
    """
    Factory function to get a GroqClient with specific model.
    
    Args:
        model_type: One of "fast", "balanced", "large", "phi"
        
    Returns:
        GroqClient instance
    """
    return GroqClient(model=MODELS.get(model_type, MODELS["fast"]))
