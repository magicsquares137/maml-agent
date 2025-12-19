from dataclasses import dataclass
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

@dataclass 
class Config:
    # Agent parameters
    max_iters: int = 50  # Match the paper's baseline
    
    # OpenAI parameters
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    service: str = "vLLM" # can set to OpenAI or TogetherAI
    togetherai_api_key: str = os.getenv("TOGETHER_AI")
    base_model: str = os.getenv("VLLM_MODEL") 
    max_tokens: int = 2000
    temperature: float = 0.0
    base_model_tokenizer: str = os.getenv("VLLM_MODEL") 
    vllm_url: str = os.getenv("VLLM_URL") 
    
    # Context management
    truncation_threshold: int = 20000  # Characters, not tokens

    # AppWorld Root
    os.environ["APPWORLD_ROOT"] = os.getenv("APPWORLD_ROOT")
    
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