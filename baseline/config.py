from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass 
class Config:
	max_iters: int = 3
	openai_api_key: str = os.getenv("OPENAI_API_KEY")
	max_tokens: int = 2000
	temperature: float = 0.0 
	base_model: str = "gpt-4o-2024-05-13"
	truncation_threshold: int = 20000