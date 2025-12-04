from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from .config import Config


# response = client.chat.completions.create(
#             model=base_model,
#             messages=[msg.dict() for msg in state.conversation_history],
#             temperature=temperature,
#             max_tokens=max_tokens
#         )

class Message(BaseModel):
    """Single conversation message"""
    role: Literal["system", "user", "assistant"]
    content: str


class AgentState(BaseModel):
	conversation_history: List[Message] = []
	iteration: int = 0
	done: bool = False

	@property 
	def should_continue(self):
		if self.iteration < Config.max_iters and not self.done:
			return True
		else:
			return False 

	def total_chars(self):
		return sum(len(msg.content) for msg in self.conversation_history)