"""
Central Groq LLM Client for Math Mentor AI
Provides unified interface for all agents
"""
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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
