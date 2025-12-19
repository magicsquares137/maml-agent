# MAML-Agent

Reinforcement learning for interactive LLM agents on the AppWorld benchmark. Implements LOOP (Leave-One-Out Proximal Policy Optimization) from [Reinforcement Learning for Long-Horizon Interactive LLM Agents](https://arxiv.org/pdf/2502.01600).

## Overview

This repository contains:
- **Baseline**: ReAct-style agent for AppWorld evaluation
- **PPO Training**: LOOP implementation for training agents with RL

The agent interacts with AppWorld's REPL environment to complete tasks by generating code, observing outputs, and iteratively working toward task completion. Note that this is a simple implementation of PPO and assumes sequential task running and no needs for sharding. Parallelization will require additional development. 

## Note on usage of AI
This project codebase was completely hand written to ensure quality and adherence to mathematical structure of PPO-LOOP and AppWorld setup, and then refined using AI (Claude, GPT5.1) to include docstrings, exception handling, and other meta-code operations. 

## Project Structure

```
maml-agent/
├── baseline/              # Baseline agent implementation
│   ├── agent.py          # ReAct agent with vLLM integration
│   ├── config.py         # Configuration management
│   ├── main.py           # Evaluation script
│   ├── models.py         # Pydantic models for agent state
│   ├── templates.py      # Prompt templates
│   └── utils.py          # Utility functions
├── ppo_baseline/         # PPO-LOOP training implementation
│   └── main.py           # PPO training loop
├── data/                 # Task data (from AppWorld)
├── experiments/          # Experiment outputs
└── requirements.txt      # Python dependencies
```

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=""           # For OpenAI models
TOGETHER_AI=""              # For Together AI models
HF_TOKEN=""                 # HuggingFace token for model access

# Paths
APPWORLD_ROOT="/path/to/maml-agent"  # Absolute path to project root

# vLLM Configuration
VLLM_MODEL="microsoft/Phi-3-mini-128k-instruct"  # Model to serve
```

### 2. Install Dependencies

**For baseline/evaluation:**
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

# With LoRA support (required for PPO training)
vllm serve microsoft/Phi-3-mini-128k-instruct \
  --port 8000 \
  --max-model-len 32768 \
  --enable-lora \
  --max-lora-rank 64
```

**Note:** For PPO training, LoRA support must be enabled in vLLM.

### 4. Configure Agent

Update `baseline/config.py` with your vLLM server URL:

```python
class Config:
    service = "vLLM"
    base_model = "microsoft/Phi-3-mini-128k-instruct"
    # Update with your vLLM server address
    vllm_url = "http://your-gpu-server:8000"
```

## Usage

### Baseline Evaluation

Run the baseline agent on AppWorld tasks:

```bash
cd baseline

# Evaluate on test set
python main.py --dataset test_normal --experiment phi3_baseline

# Evaluate on specific number of tasks
python main.py --dataset test_normal --experiment phi3_test --max-tasks 10

# Available datasets: train, dev, test_normal, test_challenge
```

**Output:**
- Task completion results
- TGC (Task Goal Correct) and SGC (Sub-Goal Correct) metrics
- Results saved to `experiments/outputs/{experiment_name}/`

### PPO Training

Train the agent with reinforcement learning:

```bash
cd ppo_baseline

python main.py
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
- Training metrics: loss, average reward, task success rate

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
- LoRA adapters for efficient training

## Key Features

### Token-Level Tracking
- Stores exact token IDs from vLLM (input and output)
- Avoids retokenization drift during training
- Enables accurate importance sampling ratios

### PPO with LoRA
- Base model stays frozen
- Only trains lightweight LoRA adapters (~50-100MB vs ~15GB full model)
- Supports iterative improvement over multiple training iterations

### ReAct Agent
- Generates reasoning + code blocks
- Executes code in AppWorld environment
- Observes results and continues iteratively
- Truncates stored outputs to avoid training on hallucinated future turns

## Results

| Model | Dataset | TGC | SGC |
|-------|---------|-----|-----|
| Phi-3 Mini (baseline) | test_normal | TBD | TBD |
| Phi-3 Mini + LOOP | test_normal | TBD | TBD |

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
- [vLLM](https://github.com/vllm-project/vllm) for fast LLM inference
- LOOP paper authors for the algorithm

## License

MIT License - see LICENSE file for details