import re
from typing import Optional, List, Tuple
from shared.models import Message


def message_parser_with_position(message: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Extract code from markdown code blocks and return code + positions.
    Returns (code, start_pos, end_pos) where positions mark the full code block including ```.
    """
    pattern = r'```(?:python)?\n(.*?)```'
    match = re.search(pattern, message, re.DOTALL)
    
    if match:
        code = match.group(1).strip()
        # match.start() is the position of the opening ```
        # match.end() is the position after the closing ```
        return code, match.start(), match.end()
    
    # Fallback for non-markdown code
    if message.strip().startswith(('print(', 'apis.', 'import ', 'from ')):
        return message.strip(), 0, len(message)
    
    return None, None, None


def render_chat_to_token_ids(messages: List[dict], tokenizer) -> List[int]:
    # messages is your list of Message objects with .role / .content
    chat = [{"role": m.role, "content": m.content} for m in messages]

    # Apply the model's chat template so tokens match what vLLM expects
    prompt_text = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return token_ids


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

# def truncate_message_history(
#     conversation_history: List[Message], 
#     threshold: int
# ) -> List[Message]:
#     """
#     Truncate conversation history following AppWorld's actual implementation:
#     1. Keep initial prompt
#     2. Keep last 5 observation blocks fully
#     3. Replace older observations with [NOT SHOWN FOR BREVITY]
#     4. If still over threshold, remove oldest (assistant, user) pairs
#     """
#     # Calculate total length
#     total_chars = sum(len(msg.content) for msg in conversation_history)
    
#     if total_chars <= threshold:
#         return conversation_history
    
#     # Keep the initial instruction message(s)
#     # Assuming first message is the system/instruction prompt
#     num_instruction_messages = 1
#     pre_messages = conversation_history[:num_instruction_messages]
#     post_messages = list(conversation_history[num_instruction_messages:])
    
#     observation_index = 0
    
#     while True:
#         # Recalculate length
#         total_chars = sum(len(msg.content) for msg in pre_messages + post_messages)
#         if total_chars <= threshold:
#             break
        
#         found_block = False
        
#         # Phase 1: Don't remove observations from last 5 blocks
#         # (5 blocks = ~2-3 turns since each turn has assistant + user)
#         if observation_index < len(post_messages) - 5:
#             # Find next observation to replace
#             for i, msg in enumerate(post_messages[observation_index:], start=observation_index):
#                 if msg.role == "user" and msg.content.startswith("Output:"):
#                     # Replace with placeholder
#                     post_messages[i] = Message(
#                         role="user",
#                         content="Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
#                     )
#                     found_block = True
#                     observation_index = i + 1
#                     break
            
#             if not found_block:
#                 observation_index = len(post_messages)
        
#         # Phase 2: If no observations left to trim, remove complete blocks
#         if not found_block and len(post_messages) > 0:
#             # Add trimmed history marker
#             first_msg = post_messages[0]
#             if not first_msg.content.endswith("[TRIMMED HISTORY]\n\n"):
#                 post_messages[0] = Message(
#                     role=first_msg.role,
#                     content=first_msg.content + "[TRIMMED HISTORY]\n\n",
#                     log_probs=first_msg.log_probs if hasattr(first_msg, 'log_probs') else None,
#                     tokenized_input=first_msg.tokenized_input if hasattr(first_msg, 'tokenized_input') else None
#                 )
            
#             # Remove 2 messages (1 assistant + 1 user)
#             if len(post_messages) >= 2:
#                 post_messages = [post_messages[0]] + post_messages[2:]
#                 found_block = True
        
#         if not found_block:
#             # Can't truncate anymore
#             break
    
#     return pre_messages + post_messages