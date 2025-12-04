# utils.py
import re
from typing import Optional, List
from models import Message

def message_parser(message: str) -> Optional[str]:
    """
    Extract code from markdown code blocks in the message.
    Returns the code string if found, None otherwise.
    """
    # Look for code blocks with ```python or just ```
    pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(pattern, message, re.DOTALL)
    
    if matches:
        # Return the first code block found
        return matches[0].strip()
    
    # If no markdown blocks, check if the entire message looks like code
    # (fallback for models that don't use markdown)
    if message.strip().startswith(('print(', 'apis.', 'import ', 'from ')):
        return message.strip()
    
    return None

def truncate_message_history(
    conversation_history: List[Message], 
    threshold: int
) -> List[Message]:
    """
    Truncate conversation history if it exceeds the character threshold.
    Keeps the system message and most recent messages.
    """
    total_chars = sum(len(msg.content) for msg in conversation_history)
    
    if total_chars <= threshold:
        return conversation_history
    
    # Always keep the first message (initial prompt with instructions)
    truncated = [conversation_history[0]]
    
    # Keep most recent messages until we're under threshold
    recent_messages = []
    current_chars = len(conversation_history[0].content)
    
    # Work backwards from most recent
    for msg in reversed(conversation_history[1:]):
        msg_chars = len(msg.content)
        if current_chars + msg_chars <= threshold:
            recent_messages.insert(0, msg)
            current_chars += msg_chars
        else:
            break
    
    truncated.extend(recent_messages)
    return truncated