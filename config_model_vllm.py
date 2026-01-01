MODEL_INFOS = [
    {
        "model_name": "deepseek-coder-33b-instruct-vllm",
        "client_name": "openai",
        "model_id": "deepseek-ai/deepseek-coder-33b-instruct",
        "model_kwargs": {
            "api_type": "chat_completions",
            # Paper greedy decoding:
            "temperature": 0,
            "top_p": 1.0,
            "seed": 100,
            "api_key_env_name": "NO_API_KEY",
            "base_url": "https://get7p2dg7934vi-8000.proxy.runpod.net/v1",

            # keep reasonable so runs don't hang forever
            "max_completion_tokens": 3000,

            # For DeepSeek, unless you *know* it supports tool calling well,
            # it's usually safer to disable tool parsing first.
            "tool_parser_name": None,
            "parallel_tool_calls": False,

            "cost_per_token": {
                "input_cache_miss": 0.0,
                "input_cache_hit": 0.0,
                "input_cache_write": 0.0,
                "output": 0.0,
            },
        },

        # Start conservative: no function-calling / tool-choice automation
        # until you confirm the baseline runs end-to-end.
        "function_calling": False,
        "tool_choice": "none",
        "function_calling_demos": False,

        # If you later enable function calling and hit schema issues with vLLM,
        # you can copy their remove_function_property_keys list here.
        "remove_function_property_keys": [
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minimum",
            "maximum",
        ],

        "model_server_config": {
            "enabled": True,
            "command": (
                "vllm serve deepseek-ai/deepseek-coder-33b-instruct "
                "--port {port} "
                "--max-model-len 8192 "
                "--gpu-memory-utilization 0.90 "
                "--max-num-seqs 4"
            ),
            "timeout": 600,
            "show_logs": False,
        },

        "part_of": ["vn", "vllm"],
        "provider": "vllm",
    },
]

import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["NO_API_KEY"] = "EMPTY"
os.environ["MODEL_SERVER_URL"] = "http://localhost:8000"

# Import libraries
from appworld import AppWorld, load_task_ids
from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent

# Load a task
task_ids = load_task_ids("test_normal")
task_id = task_ids[0]  # Get first task

print(f"Testing task: {task_id}")

# Create agent
agent = SimplifiedReActCodeAgent(
    model_config={
        "client_name": "openai",
        "api_type": "chat_completions",
        "base_url": "http://localhost:8000/v1",
        "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "api_key_env_name": "NO_API_KEY",
        "temperature": 0.0,
        "seed": 100,
        "logprobs": True,
        "top_logprobs": 1,
        "extra_body": {"return_token_ids": True},  # ADD THIS!
        "max_completion_tokens": 500,
        "cost_per_token": {
            "input_cache_hit": 0.0,
            "input_cache_miss": 0.0,
            "input_cache_write": 0.0,
            "output": 0.0
        },
        "retry_after_n_seconds": 15,
        "use_cache": False,
        "max_retries": 100,
    },
    logger_config={
        "color": True,
        "verbose": True,
    },
    prompt_file_path="experiments/prompts/react_code_agent/instructions.txt",
    ignore_multiple_calls=True,
    max_steps=5,  # Just 5 steps for quick test
    log_lm_calls=True,
)

# Initialize logger
agent.logger.initialize(
    experiment_name="test_logprobs",
    num_tasks=1,
    num_processes=1,
    process_index=0,
)

# Run the task
print("\n🚀 Running task...")
agent.solve_task(task_id)

# Check for logprobs
print("\n" + "="*60)
print("CHECKING FOR LOGPROBS")
print("="*60)

print(f"\nTotal messages: {len(agent.messages)}")

# Check each message
for i, msg in enumerate(agent.messages):
    print(f"\nMessage {i}:")
    print(f"  Role: {msg['role']}")
    
    has_logprobs = msg.get('logprobs') is not None
    print(f"  Has logprobs: {has_logprobs}")
    
    if has_logprobs:
        logprobs = msg['logprobs']
        print(f"  ✅ Num tokens: {len(logprobs)}")
        print(f"  First 3: {logprobs[:3]}")
    
    has_prompt_ids = msg.get('prompt_token_ids') is not None
    print(f"  Has prompt_token_ids: {has_prompt_ids}")

# Summary
assistant_msgs = [m for m in agent.messages if m['role'] == 'assistant']
msgs_with_logprobs = sum(1 for m in assistant_msgs if m.get('logprobs'))

print("\n" + "="*60)
print(f"✅ Assistant messages with logprobs: {msgs_with_logprobs}/{len(assistant_msgs)}")
print("="*60)


Good question! Here are your options:

## Option 1: Install Your Fork as a Dependency (Recommended)

Install your forked AppWorld directly from GitHub into your PPO repo's environment:

```bash
cd ~/path/to/your-ppo-repo
source env/bin/activate

# Uninstall the original appworld
pip uninstall appworld appworld-agents -y

# Install your fork from GitHub
pip install git+https://github.com/magicsquares137/appworld-rl.git
pip install git+https://github.com/magicsquares137/appworld-rl.git#subdirectory=experiments[simplified]
```

**Pros:** Clean separation, easy to update
**Cons:** Need to push to GitHub every time you make changes

## Option 2: Editable Install from Local Path

If both repos are on the same machine:

```bash
cd ~/path/to/your-ppo-repo
source env/bin/activate

# Install your fork in editable mode
pip install -e ~/Documents/GPT/GitHub/ariadne/appworld_fork/appworld-rl
pip install -e ~/Documents/GPT/GitHub/ariadne/appworld_fork/appworld-rl/experiments[simplified]
```

**Pros:** Changes to AppWorld fork are immediately available
**Cons:** Path-dependent, won't work on different machines

## Option 3: Git Submodule (Advanced)

Add your AppWorld fork as a submodule in your PPO repo:

```bash
cd ~/path/to/your-ppo-repo

# Add as submodule
git submodule add https://github.com/magicsquares137/appworld-rl.git appworld_fork

# Install it
pip install -e appworld_fork
pip install -e appworld_fork/experiments[simplified]
```

**Pros:** Version locked, portable, can track both repos
**Cons:** Submodules can be tricky

## Option 4: Monorepo (Simple but Messy)

Just copy the AppWorld fork into your PPO repo:

```bash
cd ~/path/to/your-ppo-repo
cp -r ~/Documents/GPT/GitHub/ariadne/appworld_fork/appworld-rl ./appworld_local

pip install -e appworld_local
pip install -e appworld_local/experiments[simplified]
```

**Pros:** Everything in one place
**Cons:** Hard to sync updates, messy git history

## My Recommendation for RunPod Setup:

Since you're working across local (laptop) and remote (RunPod), I'd suggest:

### On RunPod:
```bash
cd /workspace
git clone https://github.com/magicsquares137/appworld-rl.git
cd appworld-rl
pip install -e .
pip install -e experiments[simplified]

# Then clone your PPO repo separately
cd /workspace
git clone https://github.com/YOUR-USERNAME/your-ppo-repo.git
cd your-ppo-repo
# PPO code imports from the installed appworld package
```

### In Your PPO Code:
```python
# Just import normally - it will use your forked version
from appworld import AppWorld, load_task_ids
from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent
```

### When You Update AppWorld Fork:

**On laptop:**
```bash
cd ~/Documents/GPT/GitHub/ariadne/appworld_fork/appworld-rl
# Make changes
git commit -am "Updated logprob extraction"
git push
```

**On RunPod:**
```bash
cd /workspace/appworld-rl
git pull
# Changes are immediately available since it's editable install
```

This keeps things clean and makes it easy to work across machines. Which approach sounds best for your workflow?