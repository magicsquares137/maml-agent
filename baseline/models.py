# models.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple


class Message(BaseModel):
    """Single conversation message"""
    role: Literal["system", "user", "assistant"]
    
    # Actual generated message content (output)
    content: str
    
    # Stores, for assistant generations, a tuple
    # containing items like ("Hi", -0.081, 123)
    log_probs: Optional[List[Tuple[str, float, int]]] = None
    
    # List of tokenized ids representing input that was used to
    # generate the associated Message
    tokenized_input: Optional[List[int]] = None 


class AgentState(BaseModel):
    conversation_history: List[Message] = Field(default_factory=list)
    iteration: int = 0
    done: bool = False
    max_iters: int = 50
    
    @property 
    def should_continue(self):
        return self.iteration < self.max_iters and not self.done