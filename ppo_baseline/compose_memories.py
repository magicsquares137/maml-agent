# compose_memories.py
import argparse
import anthropic
import os
from pathlib import Path

def compose_memories(memory_paths: list, output_path: str):
    """
    Use Claude to compose multiple memory templates into one
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
            memories.append(f"### Specialist {i+1}:\n{template}")
    
    # Prompt for Claude
    prompt = f"""You have {len(memories)} specialist best practices templates from different training runs.

{chr(10).join(memories)}

Your task: Synthesize these into a single unified best practices template that:
1. Extracts common successful patterns across all specialists
2. Includes specific guidance where different specialists learned different strategies
3. Is concise (max 600 words) and highly actionable
4. Prioritizes the most impactful advice

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
    
    # Save
    with open(output_path, 'w') as f:
        f.write("# Composed Memory Template\n")
        f.write(f"# Source memories: {len(memory_paths)}\n")
        f.write("="*80 + "\n\n")
        f.write(composed)
    
    print(f"✅ Composed memory saved to {output_path}")
    print(f"   Length: {len(composed)} characters")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Compose memory templates")
    parser.add_argument("--memories", nargs="+", required=True, 
                       help="Paths to memory template files")
    parser.add_argument("--output", type=str, default="./composed_memory.txt",
                       help="Output path for composed template")
    args = parser.parse_args()
    
    # Verify files exist
    for path in args.memories:
        if not Path(path).exists():
            raise FileNotFoundError(f"Memory file not found: {path}")
    
    compose_memories(args.memories, args.output)

if __name__ == "__main__":
    main()