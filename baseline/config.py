# config.py
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass 
class Config:
    # Agent parameters
    max_iters: int = 10  # Match the paper's baseline
    
    # OpenAI parameters
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    base_model: str = "gpt-4o-2024-05-13"  # GPT-4o baseline
    max_tokens: int = 2000
    temperature: float = 0.0
    
    # Context management
    truncation_threshold: int = 20000  # Characters, not tokens
    
    @classmethod
    def for_model(cls, model_name: str):
        """Factory method for different model configs"""
        configs = {
            "gpt-4o": cls(base_model="gpt-4o-2024-05-13"),
            "gpt-4": cls(base_model="gpt-4-turbo-2024-04-09"),
            "o1": cls(
                base_model="o1-preview-2024-09-12",
                temperature=1.0,  # o1 requires temperature=1
                max_tokens=4000
            )
        }
        return configs.get(model_name, cls())