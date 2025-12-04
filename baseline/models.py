# models.py
from pydantic import BaseModel
from typing import List, Literal


class Message(BaseModel):
    """Single conversation message"""
    role: Literal["system", "user", "assistant"]
    content: str


class AgentState(BaseModel):
    conversation_history: List[Message] = []
    iteration: int = 0
    done: bool = False
    max_iters: int = 50  # Default, will be set by agent
    
    @property 
    def should_continue(self):
        return self.iteration < self.max_iters and not self.done
    
    def total_chars(self):
        return sum(len(msg.content) for msg in self.conversation_history)