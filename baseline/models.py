# models.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple


class Message(BaseModel):
    """Single conversation message"""
    role: Literal["system", "user", "assistant"]
    content: str
    log_probs: Optional[List[Tuple[str, float]]] = None # associated with output, generated tokens
    tokenized_input: Optional[List[str]] = None # associated with inputs, tokenized inputs passed into vllm


class AgentState(BaseModel):
    conversation_history: List[Message] = Field(default_factory=list)
    iteration: int = 0
    done: bool = False
    max_iters: int = 50
    
    @property 
    def should_continue(self):
        return self.iteration < self.max_iters and not self.done