import re
from typing import Optional, List, Tuple
from shared.models import Message


# def message_parser_with_position(message: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
#     """
#     Extract code from markdown code blocks and return code + positions.
#     Returns (code, start_pos, end_pos) where positions mark the full code block including ```.
#     """
#     pattern = r'```(?:python)?\n(.*?)```'
#     match = re.search(pattern, message, re.DOTALL)
    
#     if match:
#         code = match.group(1).strip()
#         # match.start() is the position of the opening ```
#         # match.end() is the position after the closing ```
#         return code, match.start(), match.end()
    
#     # Fallback for non-markdown code
#     if message.strip().startswith(('print(', 'apis.', 'import ', 'from ')):
#         return message.strip(), 0, len(message)
    
#     return None, None, None

def message_parser_with_position(
    message: str, 
    ignore_multiple_calls: bool = True
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Extract code matching AppWorld's exact logic.
    Returns (code, start_pos, end_pos) of the code block to store in history.
    """
    # Match AppWorld's regexes EXACTLY
    full_code_regex = r"```python\n(.*?)```"  # Requires 'python'
    partial_code_regex = r".*```python\n(.*)"  # For incomplete blocks
    
    output_code = ""
    text_end_pos = len(message)
    match_end = 0
    first_match_start = None
    
    # Handle complete code blocks
    for re_match in re.finditer(full_code_regex, message, re.DOTALL):
        code = re_match.group(1).strip()
        
        if first_match_start is None:
            first_match_start = re_match.start()
        
        if ignore_multiple_calls:
            # Return first block only
            return code, re_match.start(), re_match.end()
        
        output_code += code + "\n"
        match_end = re_match.end()
    
    # Check for partial code (missing closing ```)
    partial_match = re.match(partial_code_regex, message[match_end:], re.DOTALL)
    if partial_match:
        output_code += partial_match.group(1).strip()
        
        # For storage, we need to know where this partial block started
        if first_match_start is None:
            # The partial block is the first/only code
            partial_start = message.find("```python")
            if partial_start >= 0:
                first_match_start = partial_start
        
        # End position is end of message (since it's incomplete)
        return output_code.strip(), first_match_start or 0, len(message)
    
    if output_code:
        # Had complete blocks
        return output_code.strip(), first_match_start or 0, match_end
    
    # No code blocks found at all
    return None, None, None


def render_chat_to_token_ids(messages: List[Message], tokenizer) -> List[int]:
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


# def truncate_message_history(
#     conversation_history: List[Message], 
#     threshold: int
# ) -> List[Message]:
#     """
#     Truncate conversation history if it exceeds the character threshold.
#     Keeps the system message and most recent messages.
#     """
#     total_chars = sum(len(msg.content) for msg in conversation_history)
    
#     if total_chars <= threshold:
#         return conversation_history
    
#     # Always keep the first message (initial prompt with instructions)
#     truncated = [conversation_history[0]]
    
#     # Keep most recent messages until we're under threshold
#     recent_messages = []
#     current_chars = len(conversation_history[0].content)
    
#     # Work backwards from most recent
#     for msg in reversed(conversation_history[1:]):
#         msg_chars = len(msg.content)
#         if current_chars + msg_chars <= threshold:
#             recent_messages.insert(0, msg)
#             current_chars += msg_chars
#         else:
#             break
    
#     truncated.extend(recent_messages)
#     return truncated

def _shrink_output_block(msg: Message) -> Message:
    if msg.role != "user":
        return msg

    if msg.content.startswith("Output:\n```"):
        return Message(
            role=msg.role,
            content="Output:\n```\n[NOT SHOWN FOR BREVITY]\n```\n",
            log_probs=getattr(msg, "log_probs", None),
            tokenized_input=getattr(msg, "tokenized_input", None),
        )
    return msg

def truncate_message_history(
    conversation_history: list[Message],
    threshold: int,
) -> list[Message]:
    # Fast path
    if sum(len(m.content) for m in conversation_history) <= threshold:
        return conversation_history

    # Always keep the instruction
    msgs = [conversation_history[0]] + [
        Message(**m.__dict__) for m in conversation_history[1:]
    ]

    # Shrink old Output blocks first
    protect_last = 5  # AppWorld keeps recent context intact
    for i in range(1, max(1, len(msgs) - protect_last)):
        msgs[i] = _shrink_output_block(msgs[i])

    if sum(len(m.content) for m in msgs) <= threshold:
        return msgs

    # Still too long → drop oldest history blocks (but keep instruction)
    truncated = [msgs[0]]
    current = len(msgs[0].content)

    for msg in reversed(msgs[1:]):
        if current + len(msg.content) <= threshold:
            truncated.insert(1, msg)
            current += len(msg.content)
        else:
            break

    return truncated