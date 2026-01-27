#!/usr/bin/env python3
"""
Compose multiple memory templates into one unified template,
then create a full prompt file ready for use with AppWorld.

Usage:
    python compose_memories.py --memories checkpoints/specialist_d1_take2/memory_iter_3.txt checkpoints/specialist_d2/memory_iter_3.txt
    python compose_memories.py --memories mem1.txt mem2.txt --output composed_memory.txt --prompt-output final_prompt.txt
"""

import argparse
import anthropic
import os
from pathlib import Path
from datetime import datetime


def compose_memories(memory_paths: list, output_path: str) -> str:
    """
    Use Claude to compose multiple memory templates into one unified template.
    Returns the composed memory text.
    """
    print(f"\n📝 Composing {len(memory_paths)} memory templates...")
    
    # Read all memories
    memories = []
    for i, path in enumerate(memory_paths):
        with open(path, 'r') as f:
            content = f.read()
            # Extract just the template part (skip metadata lines)
            lines = content.split('\n')
            template_start = 0
            for j, line in enumerate(lines):
                if line.strip().startswith('==='):
                    template_start = j + 1
                    break
            template = '\n'.join(lines[template_start:])
            memories.append(f"### Specialist {i+1} ({Path(path).parent.name}):\n{template}")
    
    # Prompt for Claude
    prompt = f"""You have {len(memories)} specialist best practices templates from different training runs.

{chr(10).join(memories)}

Your task: Synthesize these into a single unified best practices template that:
1. Extracts common successful patterns across all specialists
2. Includes specific guidance where different specialists learned different strategies
3. Is concise (max 600 words) and highly actionable
4. Prioritizes the most impactful advice
5. Uses bullet points with **bold** headers for scannability

Write the unified template as if you're instructing an AI agent directly.

Unified Best Practices Template:
"""
    
    # Call Claude
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    composed = message.content[0].text.strip()
    
    # Save composed memory
    with open(output_path, 'w') as f:
        f.write("# Composed Memory Template\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Source memories: {[str(p) for p in memory_paths]}\n")
        f.write("=" * 80 + "\n\n")
        f.write(composed)
    
    print(f"✅ Composed memory saved to {output_path}")
    print(f"   Length: {len(composed)} characters")
    
    return composed


def create_prompt_with_memory(
    memory_text: str,
    output_path: str,
    original_prompt_path: str = "/workspace/appworld/appworld/experiments/prompts/react_code_agent/instructions.txt"
) -> str:
    """
    Create a full prompt file with memory template inserted right before task details.
    Returns the path to the created prompt file.
    """
    print(f"\n📄 Creating full prompt with memory...")
    
    # If no memory, just copy original
    if not memory_text or len(memory_text.strip()) == 0:
        print("   No memory provided, using original prompt")
        with open(original_prompt_path, 'r') as f:
            original = f.read()
        with open(output_path, 'w') as f:
            f.write(original)
        return output_path
    
    # Read original prompt
    with open(original_prompt_path, 'r') as f:
        original_prompt = f.read()
    
    # Insert right before "My name is:" which is just before the task
    marker = "My name is:"
    if marker in original_prompt:
        before_marker, after_marker = original_prompt.split(marker, 1)
        
        modified_prompt = f"""{before_marker}
**Key Patterns (learned from training)**:
{memory_text}

---

{marker}{after_marker}"""
    else:
        # Fallback: insert before the end
        modified_prompt = f"""{original_prompt}

---

**Key Patterns (learned from training)**:
{memory_text}

---
"""
    
    # Write to output
    with open(output_path, 'w') as f:
        f.write(modified_prompt)
    
    print(f"✅ Full prompt saved to {output_path}")
    print(f"   Total length: {len(modified_prompt)} characters")
    print(f"   Memory inserted before '{marker}'")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compose memory templates and create full prompt file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - compose two memory files
  python compose_memories.py --memories mem1.txt mem2.txt

  # Specify all outputs
  python compose_memories.py \\
    --memories checkpoints/specialist_d1_take2/memory_iter_3.txt \\
              checkpoints/specialist_d2/memory_iter_3.txt \\
    --output composed_memory.txt \\
    --prompt-output final_instructions.txt

  # Use custom base prompt
  python compose_memories.py \\
    --memories mem1.txt mem2.txt \\
    --base-prompt /path/to/instructions.txt
"""
    )
    parser.add_argument(
        "--memories", nargs="+", required=True,
        help="Paths to memory template files to compose"
    )
    parser.add_argument(
        "--output", type=str, default="./composed_memory.txt",
        help="Output path for composed memory template (default: ./composed_memory.txt)"
    )
    parser.add_argument(
        "--prompt-output", type=str, default="./instructions_with_memory.txt",
        help="Output path for full prompt with memory (default: ./instructions_with_memory.txt)"
    )
    parser.add_argument(
        "--base-prompt", type=str,
        default="/workspace/appworld/appworld/experiments/prompts/react_code_agent/instructions.txt",
        help="Path to original instructions.txt to modify"
    )
    parser.add_argument(
        "--memory-only", action="store_true",
        help="Only compose memories, don't create full prompt"
    )
    args = parser.parse_args()
    
    # Verify memory files exist
    for path in args.memories:
        if not Path(path).exists():
            raise FileNotFoundError(f"Memory file not found: {path}")
    
    # Verify base prompt exists (unless memory-only)
    if not args.memory_only and not Path(args.base_prompt).exists():
        raise FileNotFoundError(f"Base prompt not found: {args.base_prompt}")
    
    # Compose memories
    composed_memory = compose_memories(args.memories, args.output)
    
    # Create full prompt (unless memory-only)
    if not args.memory_only:
        create_prompt_with_memory(
            memory_text=composed_memory,
            output_path=args.prompt_output,
            original_prompt_path=args.base_prompt
        )
    
    print("\n🎉 Done!")
    print(f"   Composed memory: {args.output}")
    if not args.memory_only:
        print(f"   Full prompt:     {args.prompt_output}")


if __name__ == "__main__":
    main()