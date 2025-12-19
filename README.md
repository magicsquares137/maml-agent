# MAML-Agent

Reinforcement learning for interactive LLM agents on the AppWorld benchmark. Implements LOOP (Leave-One-Out Proximal Policy Optimization) from [Reinforcement Learning for Long-Horizon Interactive LLM Agents](https://arxiv.org/pdf/2502.01600).

## Overview

This repository contains:
- **Baseline**: ReAct-style agent for AppWorld evaluation
- **PPO Training**: LOOP implementation for training agents with RL

The agent interacts with AppWorld's REPL environment to complete tasks by generating code, observing outputs, and iteratively working toward task completion. Note that this is a simple implementation of PPO and assumes sequential task running with no parallelization. Distributed training will require additional development.

## Note on AI Usage

This project codebase was completely hand-written to ensure quality and adherence to the mathematical structure of PPO-LOOP and AppWorld setup, then refined using AI (Claude, GPT-4) to include docstrings, exception handling, and other meta-code operations.

## Project Structure

```
maml-agent/
├── .env                    # Environment variables (root level)
├── .gitignore
├── README.md
├── requirements.txt
├── shared/                 # Shared code
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py          # Loads .env here
│   ├── models.py
│   ├── templates.py
│   └── utils.py
├── baseline/              # Baseline evaluation
│   ├── __init__.py
│   └── main.py
├── ppo_baseline/          # PPO training
│   └── main.py
│   └── __init__.py
├── data/                  # AppWorld data
├── experiments/           # Results
└── checkpoints/  
```

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=""           # For OpenAI models (optional)
TOGETHER_AI=""              # For Together AI models (optional)
HF_TOKEN=""                 # HuggingFace token for model access (required)

# Paths
APPWORLD_ROOT="/path/to/maml-agent"  # Absolute path to project root

# vLLM Configuration
VLLM_MODEL="microsoft/Phi-3-mini-128k-instruct"  # Model to serve
VLLM_URL="localhost:8000/v1"
LOG_PROBS="NO" # set to NO for main baselining
```

### 2. Install Dependencies

**For baseline/evaluation:**
Note: see https://github.com/stonybrooknlp/appworld/ for installation of AppWorld and data downloads. AppWorld install should be run at directory root, and environment variable should reflect the location where ```appworld download data``` was run. Note that TogetherAI, Appworld, and vLLM have dependency conflicts, and so making separate virtual environments is recommended. 
```bash
pip install -r requirements.txt

# AppWorld compatibility fix (required)
pip install "click<8.2" --force-reinstall
```

**For vLLM server:**
```bash
pip install -r requirements_vllm.txt
```

### 3. Start vLLM Server

The agent uses a remote vLLM server for fast inference. Start the server on a GPU instance:

```bash
# For Phi-3 mini (4K context)
vllm serve microsoft/Phi-3-mini-4k-instruct --port 8000

# For Phi-3 mini (128K context) - recommended
vllm serve microsoft/Phi-3-mini-128k-instruct \
  --port 8000 \
  --max-model-len 32768  # Adjust based on GPU capacity

# With LoRA support (REQUIRED for PPO training)
vllm serve microsoft/Phi-3-mini-128k-instruct \
  --port 8000 \
  --max-model-len 32768 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-model-len 26000 
```

*NOTE*: the paper https://arxiv.org/pdf/2502.01600 uses model Qwen/Qwen2.5-32B-Instruct for tuning and baselining. This requires about 66GB VRAM to host. 

**Important:** LoRA support must be enabled in vLLM for PPO training to work. The training loop loads LoRA adapters dynamically via the vLLM API.

### 4. Configure Agent

Update `.env` with your vLLM server URL

## Usage

### Recommended: Use tmux for Long-Running Jobs

For both baseline evaluation and PPO training, it's highly recommended to use `tmux` to prevent interruptions from network disconnections:

```bash
apt update
apt install -y tmux
```

```bash
# Start a new tmux session
tmux new -s appworld

# Inside tmux, run your evaluation/training
python main.py --dataset test_normal --experiment phi3_baseline --seed 42

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t appworld

# List sessions: tmux ls
# Kill session: tmux kill-session -t appworld
```

### Baseline Evaluation

Run the baseline agent on AppWorld tasks:

```bash
# Evaluate on test set with reproducible seed
python baseline.main.py --dataset test_normal --experiment phi3_baseline --seed 42

# Evaluate on specific number of tasks
python baseline.main.py --dataset test_normal --experiment phi3_test --max-tasks 10 --seed 42

# Non-deterministic run (no seed)
python baseline.main.py --dataset test_normal --experiment phi3_baseline

# Available datasets: train, dev, test_normal, test_challenge
```

**Reproducibility:**
- Use `--seed` parameter for deterministic results
- Same seed + same model = identical outputs
- Seed is stored in result files for tracking
- Omit `--seed` for non-deterministic sampling

**Output:**
- Task completion results
- TGC (Task Goal Correct) and SGC (Sub-Goal Correct) metrics
- Results saved to `experiments/outputs/{experiment_name}/`
- Checkpoints saved every 10 tasks

### PPO Training

Train the agent with reinforcement learning:

```bash
cd ppo_baseline

# Start fresh training (recommended: use tmux)
tmux new -s ppo_training
python -m ppo_baseline.main.py --iterations 10

# Resume from crash/interruption
python -m ppo_baseline.main.py --resume ./checkpoints/checkpoint_latest.pt --iterations 10

# Quick test run
python -m ppo_baseline.main --iterations 2
```

**Training configuration** (edit in `main.py`):
```python
ppo_loop = PPO_LOOP(
    K=6,                      # Rollouts per task
    random_sample_number=40,  # Tasks sampled per iteration
    epsilon=0.2,              # PPO clip parameter
    learning_rate=1e-5,       # Learning rate
    n_epochs=3,               # Training epochs per iteration
    batch_size=8              # Minibatch size
)
```

**Training outputs:**
- LoRA checkpoints: `./checkpoints/lora_iter_{N}/`
- Full training checkpoints: `./checkpoints/checkpoint_iter_{N}.pt`
- Training metrics: `./checkpoints/training_metrics.json`
- Training curves: `./checkpoints/training_curves.png`

### Evaluate Trained LoRA

After training, evaluate a specific LoRA on the test set:

```bash
cd ppo_baseline

# Evaluate a specific iteration
python main.py --eval-only --lora-path ./checkpoints/lora_iter_5

# Evaluate final LoRA with seed for reproducibility
python main.py --eval-only --lora-path ./checkpoints/lora_iter_10 --seed 42
```

This runs the same evaluation pipeline as the baseline but with the trained LoRA adapter loaded.

## Algorithm: LOOP (Leave-One-Out PPO)

The PPO training implements Algorithm 1 from the paper:

1. **Rollout Collection**: Sample K trajectories per task from current policy
2. **Advantage Estimation**: Compute leave-one-out advantages using Equation 3
3. **Policy Update**: Update LoRA adapter using PPO with per-token importance weights (Equation 5)
4. **Iterate**: Save updated LoRA and repeat

**Key features:**
- Per-token PPO for fine-grained credit assignment
- Leave-one-out advantage estimation (no value function needed)
- Token ID tracking to avoid retokenization drift
- LoRA adapters for efficient training (~50-100MB vs ~15GB full model)
- Automatic checkpointing and crash recovery
- Training metrics tracking and visualization

## Key Features

### Token-Level Tracking
- Stores exact token IDs from vLLM (input and output)
- Avoids retokenization drift during training via `return_token_ids` API feature
- Enables accurate importance sampling ratios for PPO

### PPO with LoRA
- Base model stays frozen in memory and on vLLM server
- Only trains lightweight LoRA adapters
- Supports iterative improvement over multiple training iterations
- Full checkpoint saving/loading for crash recovery

### ReAct Agent
- Generates reasoning + code blocks
- Executes code in AppWorld environment
- Observes results and continues iteratively
- Truncates stored outputs to avoid training on hallucinated future turns

### Training Infrastructure
- **Checkpointing**: Automatic saving of training state, optimizer state, and metrics
- **Resume capability**: Continue training from any checkpoint after crashes
- **Metrics tracking**: Loss, rewards, success rates tracked per iteration
- **Visualization**: Automatic generation of training curves
- **Evaluation mode**: Test trained LoRAs on held-out test sets
- **Reproducibility**: Seed support for deterministic evaluation

## Crash Recovery

If training crashes or is interrupted:

```bash
# Training will auto-save checkpoint_latest.pt
# Resume with:
python main.py --resume ./checkpoints/checkpoint_latest.pt --iterations 10

# Or resume from specific iteration:
python main.py --resume ./checkpoints/checkpoint_iter_5.pt --iterations 5
```

All training state (iteration number, LoRA path, optimizer state, metrics) is preserved.

## Best Practices

### Long-Running Jobs
- **Always use tmux** for evaluation and training
- Prevents loss of progress from SSH disconnections
- Allows monitoring progress by reattaching to session

### Reproducibility
- Use `--seed` parameter for baseline evaluation
- Document seeds in experiment names (e.g., `phi3_baseline_seed42`)
- Store seeds with results for future reference

### Checkpointing
- Baseline auto-saves every 10 tasks
- PPO auto-saves after each iteration
- Both can be safely interrupted and resumed

## Results

| Model | Dataset | TGC | SGC | Seed |
|-------|---------|-----|-----|------|
| Phi-3 Mini (baseline) | test_normal | TBD | TBD | 42 |
| Phi-3 Mini + LOOP (iter 5) | test_normal | TBD | TBD | 42 |
| Phi-3 Mini + LOOP (iter 10) | test_normal | TBD | TBD | 42 |

## Known Limitations

- **Sequential execution**: No parallelization of rollout collection
- **vLLM dependency**: Requires LoRA-enabled vLLM server for training
- **Memory requirements**: Policy model loaded in memory during training (~15GB for Phi-3)
- **Single GPU**: No multi-GPU distribution support

## Citation

If you use this code, please cite:

```bibtex
@article{zhao2025reinforcement,
  title={Reinforcement Learning for Long-Horizon Interactive LLM Agents},
  author={Zhao, Wenhao and others},
  journal={arXiv preprint arXiv:2502.01600},
  year={2025}
}
```

## Acknowledgments

- [AppWorld](https://appworld.dev/) benchmark for interactive agent evaluation
- [vLLM](https://github.com/vllm-project/vllm) for fast LLM inference with LoRA support
- LOOP paper authors for the algorithm
- Agent Lightning team for `return_token_ids` API feature

## License

MIT License - see LICENSE file for details