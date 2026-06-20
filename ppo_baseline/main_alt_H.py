import os
from dotenv import load_dotenv, find_dotenv
# IMPORTANT: load .env and set APPWORLD_ROOT BEFORE importing appworld below.
# appworld resolves APPWORLD_ROOT at import time and defaults it to the current
# working directory if unset, so loading the .env afterwards is too late.
load_dotenv(find_dotenv())
_appworld_root = os.getenv("APPWORLD_ROOT")
if _appworld_root:
    os.environ["APPWORLD_ROOT"] = _appworld_root

# NOTE: This is the ORCHESTRATOR process. It must NEVER import torch /
# transformers / peft — appworld's native stack and torch crash (segfault in GC)
# when co-resident in one process. The PPO weight update runs in a separate
# process via ppo_baseline/train_step.py (invoked with the vllm_env python).
import sys
import random
import uuid
from appworld import AppWorld, load_task_ids
from shared.config import Config
from shared.agent import ReactAgent
from shared.models import AgentState, Message
from typing import Dict, Optional, List, Union, Dict
from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent
from pathlib import Path
from tqdm import tqdm
import subprocess
import time
import signal
import requests
import json
import pickle
# matplotlib and anthropic are imported lazily where used so the appworld env
# doesn't need them installed (plotting / --use-memory are optional).

def _convert_to_agent_state(appworld_agent):
	"""Module-level (picklable) twin of PPO_LOOP.convert_to_agent_state, used by
	the rollout worker so it can run inside a multiprocessing pool."""
	agent_state = AgentState(max_iters=appworld_agent.max_steps)
	for msg in appworld_agent.messages:
		if msg["role"] == "assistant" and msg.get("logprobs"):
			agent_state.conversation_history.append(Message(
				role=msg["role"], content=msg["content"],
				log_probs=msg["logprobs"], tokenized_input=msg.get("prompt_token_ids")))
		elif msg["role"] == "user":
			agent_state.conversation_history.append(Message(role=msg["role"], content=msg["content"]))
	agent_state.iteration = appworld_agent.step_number
	return agent_state


def _rollout_worker(unit: dict) -> dict:
	"""Run ONE (task, rollout) unit, fully isolated. Module-level so it is
	picklable for multiprocessing. Uses a UNIQUE experiment_name so concurrent
	rollouts of the same task get separate AppWorld DBs (no collision). Returns
	the exact same result-dict shape as the original sequential loop.

	NOTE: only the experiment_name (where the DB/logs are written) differs from
	the old inline loop — the agent config, sampling, seed, and outcome are
	identical, so a rollout's result is unchanged by parallelization.
	"""
	import os
	import uuid as _uuid
	from appworld import AppWorld
	from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent

	os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
	os.environ.setdefault("NO_API_KEY", "EMPTY")
	# The agent's fill_model_server_url() always reads MODEL_SERVER_URL even when
	# base_url has no template (ours doesn't), so it just needs to exist. Set it
	# here so the worker is self-sufficient (spawned workers / non-main entry).
	os.environ.setdefault("MODEL_SERVER_URL", unit["vllm_url"])

	expname = unit["experiment_name"]
	agent = SimplifiedReActCodeAgent(
		model_config={
			"client_name": "openai",
			"api_type": "chat_completions",
			"base_url": unit["vllm_url"],
			"name": unit["model_name"],
			"api_key_env_name": "NO_API_KEY",
			"temperature": unit["temperature"],
			"seed": unit["seed"],
			"logprobs": True,
			"top_logprobs": 1,
			"extra_body": {"return_token_ids": True},
			"max_completion_tokens": unit["max_tokens"],
			"cost_per_token": {
				"input_cache_hit": 0.0, "input_cache_miss": 0.0,
				"input_cache_write": 0.0, "output": 0.0,
			},
			"retry_after_n_seconds": 15,
			"use_cache": False,
			"max_retries": 100,
		},
		logger_config={"color": False, "verbose": False},
		appworld_config={"random_seed": 100},
		prompt_file_path=unit["prompt_path"],
		ignore_multiple_calls=True,
		max_prompt_length=None,
		max_output_length=None,
		max_steps=unit["train_max_iters"],
	)

	result = {
		"task_id": unit["task_id"], "completed": False, "iterations": 0,
		"error": None, "conversation_length": 0, "overall_success": None,
		"uuid": _uuid.uuid4(), "agent_state": None, "evaluation_details": None,
	}
	try:
		agent.logger.initialize(expname, 1, 1, 0)
		# Unique experiment_name -> isolated DB for this rollout.
		with AppWorld.initializer(update_defaults=True, experiment_name=expname, random_seed=100):
			agent.solve_task(unit["task_id"])
			evaluation = agent.world.evaluate().to_dict()
		completed = evaluation["success"]
		overall_success = len(evaluation["passes"]) / evaluation["num_tests"]
		agent_state = _convert_to_agent_state(agent)
		agent_state.done = completed
		result.update(
			completed=completed, iterations=agent.step_number,
			conversation_length=len(agent_state.conversation_history),
			overall_success=overall_success, evaluation_details=evaluation,
			agent_state=agent_state,
		)
	except Exception as e:
		import traceback as _tb
		result["error"] = _tb.format_exc()
		result["agent_state"] = None
	finally:
		del agent
	return result


class PPO_LOOP:
	def __init__(
		self, 
		# Defaults are from paper https://arxiv.org/pdf/2502.01600
		K: int = 6, 
		random_sample_number: int = 40, 
		difficulties: List[int] = [1,2], 
		sets: List[str] = ["train", "dev"], 
		config: Config = None,
		epsilon: float = 0.2,  # PPO clip parameter
		learning_rate: float = 1e-5,
		n_epochs: int = 3,
		batch_size: int = 8,
		checkpoint_dir: str = "./checkpoints",
		resume_from: str = None,
		anthropic_api_key: str = None,
		# --- rollout sampling (LOOP needs DIVERSE trajectories per task) ---
		rollout_temperature: float = 1.0,   # paper: temperature 1.0
		rollout_seed_base: int = 1000,
		train_max_iters: int = 40,          # paper: <=40 interactions during training
		rollout_workers: int = 1,           # 1 = sequential; >1 = parallel processes

		# --- vLLM serving (rollout phase) ---
		vllm_tensor_parallel: int = 2,
		vllm_gpu_mem_util: float = 0.90,
		vllm_max_model_len: int = 20000,
		vllm_max_num_seqs: int = 8,   # rollouts are sequential; small batch fits 16GB
		vllm_quantization: str = None,
		# --- training phase (runs in a separate process: train_step.py) ---
		load_in_4bit: bool = False,
		lora_r: int = 16,
		lora_alpha: int = 32,
		lora_dropout: float = 0.05,
		lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
		trainer_python: str = None,   # python that has torch/transformers/peft (vllm_env)
		vllm_bin: str = None,         # path to the `vllm` CLI (vllm_env/bin/vllm)
		# --- portability ---
		prompt_file_path: str = None,
		# --- prompt-memory (H) feature: requires Anthropic API; off = pure RL ---
		use_memory: bool = False,
	) -> None:

		# Set up storage and stats dir
		self.checkpoint_dir = Path(checkpoint_dir)
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		
		# Training metrics tracking
		self.training_history = {
			"iterations": [],
			"avg_rewards": [],
			"success_rates": [],
			"avg_losses": [],
			"completed_tasks": []
		}

		self.use_memory = use_memory
		if self.use_memory:
			api_key = os.getenv("ANTHROPIC_API_KEY")
			if not api_key:
				raise ValueError("--use-memory requires ANTHROPIC_API_KEY to be set")
			self.anthropic_client = anthropic.Anthropic(api_key=api_key)
			print(f"✅ Anthropic client initialized (prompt-memory ON)")
		else:
			self.anthropic_client = None
			print("ℹ️  Prompt-memory (H) disabled — pure RL run")

		# initialize prompt injection template as empty
		self.H = ""
		
		# Track memory evolution
		self.memory_history = []

		# Start with None - first rollouts use base model
		self.current_lora_path = None
		
		# Will initialize after first rollout collection
		self.policy_model = None
		self.optimizer = None
		
		# Resume from checkpoint if specified
		if resume_from:
			self.load_checkpoint(resume_from)

		# Number of rollouts per episode
		self.K = K

		# Number of tasks to be sampled from task set during rollouts
		self.random_sample_number = random_sample_number
		
		# [1, 2] -> correspond to task difficulty in task set
		self.difficulties = difficulties

		# ["id_123", "id_345"]
		temp_train_ids = [
			tid
			for dataset_name in sets
			for tid in load_task_ids(dataset_name)
		] 

		# filter by requested difficulty
		self.train_ids = [] # <- ["123", "456", ...]
		i = 1
		for x in temp_train_ids:
			print(f"Init difficulty check task {i} of {len(temp_train_ids)}")
			temp_world = AppWorld(task_id=x)
			if temp_world.task.ground_truth.metadata["difficulty"] in difficulties:
				self.train_ids.append(x)
			temp_world.close()
			i += 1

		# note: on iteration 1, we will use model with no lora and then update lora 
		# then on all subsequent iterations, we will applied loras on top of base
		self.config = config
		self.epsilon = epsilon
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.iteration = 0  # Track current iteration
		
		# LoRA hyperparameters (passed to the trainer subprocess; the peft
		# LoraConfig itself is built in train_step.py, which owns torch/peft).
		self.lora_r = lora_r
		self.lora_alpha = lora_alpha
		self.lora_dropout = lora_dropout
		self.lora_target_modules = lora_target_modules

		# Trainer subprocess config: which python has torch/transformers/peft,
		# and where the vllm CLI lives (orchestrator may run in a torch-free env).
		self.trainer_python = (
			trainer_python
			or os.getenv("TRAINER_PYTHON")
			or "/home/smcclendon/Documents/github/maml-agent/vllm_env/bin/python"
		)
		self.vllm_bin = (
			vllm_bin
			or os.getenv("VLLM_BIN")
			or "/home/smcclendon/Documents/github/maml-agent/vllm_env/bin/vllm"
		)

		self.vllm_process = None
		self.vllm_port = 8000
		self.vllm_host = "localhost"

		# Rollout sampling config (diverse trajectories -> nonzero LOOP advantage)
		self.rollout_temperature = rollout_temperature
		self.rollout_seed_base = rollout_seed_base
		self.train_max_iters = train_max_iters
		self.rollout_workers = rollout_workers

		# vLLM serving config
		self.vllm_tensor_parallel = vllm_tensor_parallel
		self.vllm_gpu_mem_util = vllm_gpu_mem_util
		self.vllm_max_model_len = vllm_max_model_len
		self.vllm_max_num_seqs = vllm_max_num_seqs
		self.vllm_quantization = vllm_quantization

		# Training config
		self.load_in_4bit = load_in_4bit

		# Prompt file: default to local appworld prompt, overridable via env/arg
		self.prompt_file_path = (
			prompt_file_path
			or os.getenv("APPWORLD_PROMPT_FILE")
			or "/home/smcclendon/Documents/github/appworld/appworld-rl/experiments/prompts/react_code_agent/instructions.txt"
		)

	def start_vllm_server(self, lora_path: str | None = None) -> bool:
		print("\n🚀 Starting vLLM server...")

		cmd = [
			self.vllm_bin, "serve", self.config.base_model,
			"--host", self.vllm_host,                 # important if not localhost
			"--port", str(self.vllm_port),
			"--max-model-len", str(self.vllm_max_model_len),
			"--gpu-memory-utilization", str(self.vllm_gpu_mem_util),
			"--tensor-parallel-size", str(self.vllm_tensor_parallel),
			"--max-num-seqs", str(self.vllm_max_num_seqs),
			"--enable-lora",
			"--max-loras", "2",
			"--max-lora-rank", "64",
		]

		if self.vllm_quantization:
			cmd += ["--quantization", self.vllm_quantization]
			print(f"   Quantization: {self.vllm_quantization}")

		if lora_path:
			cmd += ["--lora-modules", f"ppo_adapter={lora_path}"]
			print(f"   Loading LoRA: {lora_path}")
		else:
			print("   Loading base model (no LoRA)")

		env = os.environ.copy()

		# Log to file to avoid PIPE deadlock
		log_path = getattr(self, "vllm_log_path", "/tmp/vllm_server.log")
		log_f = open(log_path, "ab", buffering=0)

		self.vllm_process = subprocess.Popen(
			cmd,
			stdout=log_f,
			stderr=log_f,
			env=env,
			start_new_session=True,   # lets you kill the whole process group cleanly
		)

		print(f"   PID: {self.vllm_process.pid}")
		print(f"   Logs: {log_path}")
		print("   Waiting for vLLM to start...", end="", flush=True)

		max_wait_time = 3600
		base = f"http://{self.vllm_host}:{self.vllm_port}"

		for i in range(max_wait_time):
			# If process exited, show logs tail and fail fast
			rc = self.vllm_process.poll()
			if rc is not None:
				print(f"\n ❌ vLLM exited early (return code {rc}).")
				try:
					# show last ~200 lines
					tail = subprocess.check_output(["bash", "-lc", f"tail -n 200 {log_path}"], text=True)
					print("---- vLLM log tail ----")
					print(tail)
					print("-----------------------")
				except Exception:
					pass
				return False

			try:
				# /v1/models tends to be a reliable readiness check for vLLM OpenAI server
				r = requests.get(f"{base}/v1/models", timeout=1)
				if r.status_code == 200:
					print(" ✅ Ready!")
					time.sleep(1)
					return True
			except requests.RequestException:
				pass

			time.sleep(1)
			if i % 10 == 0 and i > 0:
				print(".", end="", flush=True)

		print("\n ❌ Timed out waiting for vLLM.")
		return False
	
	def stop_vllm_server(self):
		"""Stop vLLM server and free GPU memory"""
		if self.vllm_process is None:
			print("\n⚠️  No vLLM process to stop")
			return
		
		print("\n🛑 Stopping vLLM server...")
		
		try:
			# Get the PID before killing
			vllm_pid = self.vllm_process.pid
			print(f"   vLLM main PID: {vllm_pid}")
			
			# Kill the process group (this kills all child processes too)
			try:
				os.killpg(os.getpgid(vllm_pid), signal.SIGTERM)
				print("   Sent SIGTERM to process group")
			except ProcessLookupError:
				print("   Process already dead")
			
			# Wait a bit
			time.sleep(3)
			
			# Force kill if still alive
			try:
				os.killpg(os.getpgid(vllm_pid), signal.SIGKILL)
				print("   Sent SIGKILL to process group")
			except ProcessLookupError:
				pass
			
		except Exception as e:
			print(f"   ⚠️  Error stopping vLLM: {e}")
		
		finally:
			self.vllm_process = None
			
			# Nuclear option: kill ALL vLLM and Ray processes. Best-effort — a
			# missing binary (e.g. `ray` is in vllm_env, NOT the appworld/orchestrator
			# env) must NOT crash cleanup, or it kills the whole run after rollouts.
			print("   Cleaning up vLLM/Ray processes...")
			for _cmd in (["pkill", "-9", "-f", "vllm"],
						 ["pkill", "-9", "-f", "ray::"],
						 ["pkill", "-9", "-f", "_raylet"],
						 ["ray", "stop", "--force"]):
				try:
					subprocess.run(_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
				except FileNotFoundError:
					pass  # binary not on PATH in this env

			# Also kill Ray completely (if importable in this env)
			try:
				import ray
				if ray.is_initialized():
					ray.shutdown()
			except Exception:
				pass
			
			# Close log files if they exist
			if hasattr(self, 'vllm_stdout_file'):
				self.vllm_stdout_file.close()
			if hasattr(self, 'vllm_stderr_file'):
				self.vllm_stderr_file.close()
			
			# Wait for GPU memory to actually be freed (the vLLM process group is
			# killed above; GPU memory is released by the OS as it exits).
			print("   Waiting for GPU cleanup...", end="", flush=True)
			time.sleep(10)
			print(" Done")

	def save_checkpoint(self):
		"""Save orchestrator checkpoint (no torch state here; optimizer state
		lives next to the LoRA at <current_lora_path>/optimizer.pt)."""
		checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}.pkl"

		checkpoint = {
			"iteration": self.iteration,
			"current_lora_path": self.current_lora_path,
			"training_history": self.training_history,
			"H": self.H,
			"config": {
				"K": self.K,
				"random_sample_number": self.random_sample_number,
				"epsilon": self.epsilon,
				"learning_rate": self.learning_rate,
				"n_epochs": self.n_epochs,
				"batch_size": self.batch_size
			}
		}

		with open(checkpoint_path, "wb") as f:
			pickle.dump(checkpoint, f)
		print(f"💾 Saved checkpoint to {checkpoint_path}")

		# Also save a "latest" checkpoint
		with open(self.checkpoint_dir / "checkpoint_latest.pkl", "wb") as f:
			pickle.dump(checkpoint, f)

	def load_checkpoint(self, checkpoint_path: str):
		"""Load orchestrator checkpoint to resume"""
		print(f"📂 Loading checkpoint from {checkpoint_path}")

		with open(checkpoint_path, "rb") as f:
			checkpoint = pickle.load(f)

		self.iteration = checkpoint["iteration"]
		self.current_lora_path = checkpoint["current_lora_path"]
		self.training_history = checkpoint["training_history"]
		self.H = checkpoint.get("H", "")  # Load memory template
		self.memory_history = checkpoint.get("memory_history", [])
		
		print(f"✅ Resumed from iteration {self.iteration}")
		print(f"   Current LoRA: {self.current_lora_path}")

	def _save_memory_snapshot(self):
		"""Save current memory template"""
		memory_path = self.checkpoint_dir / f"memory_iter_{self.iteration}.txt"
		with open(memory_path, 'w') as f:
			f.write(f"# Memory Template - Iteration {self.iteration}\n")
			f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write("="*80 + "\n\n")
			f.write(self.H)
		
		print(f"   💾 Saved memory to {memory_path}")

	def save_metrics(self):
		"""Save training metrics to JSON"""
		metrics_path = self.checkpoint_dir / "training_metrics.json"
		
		with open(metrics_path, 'w') as f:
			json.dump(self.training_history, f, indent=2)
		
		print(f"📊 Saved metrics to {metrics_path}")
	
	def plot_training_curves(self):
		"""Generate training curves"""
		if not self.training_history["iterations"]:
			return

		try:
			import matplotlib
			matplotlib.use("Agg")
			import matplotlib.pyplot as plt
		except Exception as e:
			print(f"⚠️  matplotlib unavailable, skipping plots: {e}")
			return

		fig, axes = plt.subplots(2, 2, figsize=(15, 10))
		
		# Average Reward
		axes[0, 0].plot(self.training_history["iterations"], 
					   self.training_history["avg_rewards"], 
					   'b-', marker='o')
		axes[0, 0].set_xlabel('Iteration')
		axes[0, 0].set_ylabel('Average Reward')
		axes[0, 0].set_title('Average Reward over Training')
		axes[0, 0].grid(True)
		
		# Success Rate
		axes[0, 1].plot(self.training_history["iterations"], 
					   self.training_history["success_rates"], 
					   'g-', marker='o')
		axes[0, 1].set_xlabel('Iteration')
		axes[0, 1].set_ylabel('Success Rate (%)')
		axes[0, 1].set_title('Task Success Rate over Training')
		axes[0, 1].grid(True)
		
		# Average Loss
		axes[1, 0].plot(self.training_history["iterations"], 
					   self.training_history["avg_losses"], 
					   'r-', marker='o')
		axes[1, 0].set_xlabel('Iteration')
		axes[1, 0].set_ylabel('Average Loss')
		axes[1, 0].set_title('Training Loss over Iterations')
		axes[1, 0].grid(True)
		
		# Completed Tasks
		axes[1, 1].plot(self.training_history["iterations"], 
					   self.training_history["completed_tasks"], 
					   'm-', marker='o')
		axes[1, 1].set_xlabel('Iteration')
		axes[1, 1].set_ylabel('Completed Tasks')
		axes[1, 1].set_title('Completed Tasks per Iteration')
		axes[1, 1].grid(True)
		
		plt.tight_layout()
		
		# Save figure
		fig_path = self.checkpoint_dir / "training_curves.png"
		plt.savefig(fig_path, dpi=150, bbox_inches='tight')
		print(f"📈 Saved training curves to {fig_path}")
		plt.close()

	def _create_prompt_with_memory(self) -> str:
		"""
		Create temporary prompt file with memory template inserted right before task details
		"""
		original_prompt_path = self.prompt_file_path
		
		# If no memory yet (iteration 0), use original
		if not self.H or len(self.H.strip()) == 0:
			print("   Using original prompt (no memory yet)")
			return original_prompt_path
		
		# Read original prompt
		with open(original_prompt_path, 'r') as f:
			original_prompt = f.read()
		
		# Insert right before "My name is:" which is just before the task
		marker = "My name is:"
		if marker in original_prompt:
			before_marker, after_marker = original_prompt.split(marker, 1)
			
			modified_prompt = f"""{before_marker}
	**Key Patterns (learned from training)**:
	{self.H}

	---

	{marker}{after_marker}"""
		else:
			# Fallback: just append before the end
			modified_prompt = f"""{original_prompt}

	---
	**Key Patterns**:
	{self.H}

	---
	"""
		
		# Write to temp file
		temp_path = self.checkpoint_dir / f"prompt_with_memory_iter_{self.iteration}.txt"
		with open(temp_path, 'w') as f:
			f.write(modified_prompt)
		
		print(f"   Using modified prompt with memory ({len(self.H)} chars)")
		return str(temp_path)

	def collect_rollouts(self) -> List[dict]:
		from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent
		
		try:
			response = requests.get(
				f"http://{self.vllm_host}:{self.vllm_port}/health",
				timeout=2
			)
			if response.status_code != 200:
				raise RuntimeError("vLLM server not healthy!")
		except requests.exceptions.RequestException as e:
			raise RuntimeError(f"vLLM server not running! {e}")
		
		task_set = random.sample(self.train_ids, self.random_sample_number)

		# Create prompt with memory injected (once, shared by all rollouts)
		temp_prompt_path = self._create_prompt_with_memory()

		# vLLM selects a served LoRA via the request `model` field, NOT via
		# extra_body. When an adapter is active the model name must be the served
		# adapter name; otherwise rollouts silently run on the BASE model.
		model_name = "ppo_adapter" if self.current_lora_path else self.config.base_model

		# Build one work unit per (task, rollout). Each gets a UNIQUE
		# experiment_name so concurrent rollouts of the same task use isolated
		# AppWorld DBs. Unique per-rollout seed keeps the K trajectories diverse
		# (so LOOP advantages are nonzero).
		units = []
		for index, task_id in enumerate(task_set):
			for rollout in range(self.K):
				units.append({
					"task_id": task_id,
					"experiment_name": f"ppo_i{self.iteration}_t{index}_r{rollout}",
					"model_name": model_name,
					"vllm_url": self.config.vllm_url,
					"temperature": self.rollout_temperature,
					"seed": self.rollout_seed_base + index * self.K + rollout,
					"max_tokens": self.config.max_tokens,
					"train_max_iters": self.train_max_iters,
					"prompt_path": temp_prompt_path,
				})

		workers = max(1, int(getattr(self, "rollout_workers", 1)))
		if workers == 1:
			# Sequential path (default) — identical rollout logic to before.
			all_rollouts = [
				_rollout_worker(u)
				for u in tqdm(units, desc="Rollouts (sequential)")
			]
		else:
			# Parallel path — N isolated worker processes hitting the shared vLLM
			# server (which batches the concurrent requests). spawn gives each
			# worker a clean interpreter (no inherited AppWorld global state).
			import multiprocessing as mp
			print(f"\n⚡ Collecting {len(units)} rollouts with {workers} parallel workers "
				  f"(vLLM batches them; max_num_seqs={self.vllm_max_num_seqs})")
			ctx = mp.get_context("spawn")
			all_rollouts = []
			with ctx.Pool(processes=workers) as pool:
				for r in tqdm(pool.imap_unordered(_rollout_worker, units),
							  total=len(units), desc=f"Rollouts (x{workers})"):
					all_rollouts.append(r)

		# Visibility: how many rollouts have usable data vs errored (the worker
		# otherwise swallows per-rollout exceptions into the result dict).
		n_ok = sum(1 for r in all_rollouts if r.get("agent_state") is not None)
		n_err = sum(1 for r in all_rollouts if r.get("error"))
		n_logprobs = sum(
			1 for r in all_rollouts
			if r.get("agent_state") is not None
			and any(m.role == "assistant" and m.log_probs for m in r["agent_state"].conversation_history)
		)
		print(f"   rollout summary: {len(all_rollouts)} total | {n_ok} with agent_state | "
			  f"{n_logprobs} with logprobs | {n_err} errored")
		if n_err:
			for r in all_rollouts:
				if r.get("error"):
					print("   ---- sample rollout traceback ----")
					print(str(r['error']))
					print("   ----------------------------------")
					break

		return all_rollouts, task_set

	def convert_to_agent_state(self, appworld_agent) -> AgentState:
		agent_state = AgentState(max_iters=appworld_agent.max_steps)
		
		for msg in appworld_agent.messages:
			if msg["role"] == "assistant" and msg.get("logprobs"):
				pydantic_msg = Message(
					role=msg["role"],
					content=msg["content"],
					log_probs=msg["logprobs"],
					tokenized_input=msg.get("prompt_token_ids")
				)
				agent_state.conversation_history.append(pydantic_msg)
			elif msg["role"] == "user":
				pydantic_msg = Message(
					role=msg["role"],
					content=msg["content"]
				)
				agent_state.conversation_history.append(pydantic_msg)
		
		agent_state.iteration = appworld_agent.step_number
		# DON'T access agent.world here - it might be closed!
		# agent_state.done will be set by the caller
		
		return agent_state

	def get_advantages(
		self, 
		all_rollouts: List[dict], 
		task_set: List
	) -> List[dict]:
		"""
		Compute leave-one-out advantages using Equation 3 from the paper:
		A(c, x_k) = (K/(K-1)) * (R(c, x_k) - (1/K) * sum_i R(c, x_i))
		
		This is mathematically equivalent to:
		A(c, x_k) = R(c, x_k) - (1/(K-1)) * sum_{i!=k} R(c, x_i)
		"""
		updated_rollouts = []
		
		for task in task_set:
			# Get all K rollouts for this task
			task_rollouts = [x for x in all_rollouts if x["task_id"] == task]

			if len(task_rollouts) != self.K:
				raise Exception(
					f"Expected {self.K} rollouts for task {task}, but found {len(task_rollouts)}. "
					f"Check if collect_rollouts() completed successfully."
				)
			
			# Compute average reward across ALL K rollouts
			avg_reward = sum(x.get("overall_success", 0) or 0 for x in task_rollouts) / self.K
			
			for rollout in task_rollouts:
				rollout_reward = rollout.get("overall_success", 0) or 0
				
				# Equation 3: A(c, x_k) = (K/(K-1)) * (R(c, x_k) - avg_reward)
				advantage = (self.K / (self.K - 1)) * (rollout_reward - avg_reward)
				
				rollout["advantage"] = advantage
				updated_rollouts.append(rollout)
		
		if len(updated_rollouts) != len(all_rollouts):
			raise Exception(f"Missing rollouts during advantage calculation: "
						   f"expected {len(all_rollouts)}, got {len(updated_rollouts)}")
		
		return updated_rollouts

	def _update_memory_template(self, rollouts: List[dict]) -> str:
		"""
		Generate best practices template by feeding all conversation history to Claude
		"""
		if not self.anthropic_client:
			print("   ⚠️  No Anthropic client - skipping memory update")
			return self.H
		
		print("\n📝 Generating best practices template from rollouts...")
		
		# Build the input for Claude
		conversations_text = self._format_rollouts_for_llm(rollouts)
		
		# Call Claude to synthesize best practices
		prompt = f"""You are analyzing conversation logs from an AI agent attempting to solve AppWorld tasks.

	Here are {len(rollouts)} task execution logs from iteration {self.iteration}:

	{conversations_text}

	Previous best practices from iteration {self.iteration - 1}:
	{self.H if self.H else "[None - this is the first iteration]"}

	Your task: Create a CONCISE checklist (max 150 words, 5-7 bullet points) of the most impactful patterns.

	Format as short bullets:
	- [Pattern]: [One sentence]

	Focus on HIGH-IMPACT patterns that directly prevent failures or improve success rate.
	Examples:
	- Authentication: Check login requirements before accessing user-specific data
	- Loop avoidance: If action fails 2x, try different approach
	- Persistence: Attempt 8+ actions before giving up

	Checklist:
	"""
		
		try:
			new_H = self._call_claude_api(prompt)
			print(f"   ✅ Generated template ({len(new_H)} characters)")
			return new_H
		except Exception as e:
			print(f"   ❌ Failed to generate template: {e}")
			print(f"   Keeping previous template")
			#return self.H
			raise Exception(f"Unable to call Claude: {e}")

	def _format_rollouts_for_llm(self, rollouts: List[dict]) -> str:
	    """
	    Format rollout conversations into text for LLM analysis
	    Sample strategically to stay under token limits
	    """
	    formatted = []
	    
	    # Separate successes and failures
	    successful = [r for r in rollouts if r.get("completed", False)]
	    failed = [r for r in rollouts if not r.get("completed", False)]
	    
	    # Sample: 5 successes + 10 failures (failures are more informative)
	    sampled_success = successful[:2] if len(successful) > 2 else successful
	    sampled_failed = failed[:6] if len(failed) > 6 else failed
	    
	    sampled = sampled_success + sampled_failed
	    
	    print(f"   Sampling {len(sampled)} rollouts ({len(sampled_success)} success, {len(sampled_failed)} failed)")
	    
	    for i, rollout in enumerate(sampled):
	        agent_state = rollout.get("agent_state")
	        if not agent_state:
	            continue
	        
	        # Header for this rollout
	        success = "SUCCESS" if rollout.get("completed", False) else "FAILED"
	        task_id = rollout.get("task_id", "unknown")
	        steps = rollout.get("iterations", 0)
	        score = rollout.get("overall_success", 0)
	        
	        rollout_text = [
	            f"\n{'='*60}",
	            f"Rollout {i+1}/{len(sampled)}: {success}",
	            f"Task: {task_id}",
	            f"Steps: {steps}",
	            f"Score: {score:.2f}",
	            f"{'='*60}\n"
	        ]
	        
	        # TRUNCATE conversation to first 10 messages only (most important part)
	        conversation = agent_state.conversation_history[:10]
	        
	        for msg in conversation:
	            # Also truncate individual messages to 300 chars
	            content = msg.content[:300] if len(msg.content) > 300 else msg.content
	            rollout_text.append(f"{msg.role.upper()}: {content}\n")
	        
	        if len(agent_state.conversation_history) > 10:
	            rollout_text.append(f"... [{len(agent_state.conversation_history) - 10} more messages truncated]\n")
	        
	        formatted.append("\n".join(rollout_text))
	    
	    return "\n\n".join(formatted)

	def _call_claude_api(self, prompt: str) -> str:
		"""
		Call Claude API to generate the memory template
		"""
		import anthropic
		
		# Get API key from environment
		api_key = os.environ.get("ANTHROPIC_API_KEY")
		if not api_key:
			raise ValueError("ANTHROPIC_API_KEY not set in environment")
		
		client = anthropic.Anthropic(api_key=api_key)
		
		try:
			message = client.messages.create(
				model="claude-sonnet-4-5-20250929",
				max_tokens=2000,
				temperature=0.7,
				messages=[
					{"role": "user", "content": prompt}
				]
			)
			
			# Extract text from response
			new_H = message.content[0].text.strip()
			return new_H
			
		except Exception as e:
			print(f"   ⚠️  Claude API call failed: {e}")
			print(f"   Keeping previous template")
			raise ValueError(f"Claude not called: {e}") 	

	def _serialize_rollouts_for_training(self, updated_rollouts):
		"""Convert rollouts to plain dicts the trainer can load without appworld.
		Each assistant message carries the exact vLLM token ids + old logprobs."""
		train_data = []
		for r in updated_rollouts:
			if r.get("agent_state") is None or r.get("error") is not None:
				continue
			messages = []
			for m in r["agent_state"].conversation_history:
				if m.role != "assistant":
					continue
				if not m.log_probs or not m.tokenized_input:
					continue
				messages.append({
					"tokenized_input": list(m.tokenized_input),
					# log_probs entries: (token_str, old_logprob, token_id)
					"log_probs": [tuple(x) for x in m.log_probs],
				})
			if messages:
				train_data.append({"advantage": r.get("advantage", 0), "messages": messages})
		return train_data

	def _run_training_subprocess(self, updated_rollouts) -> float:
		"""Pickle rollouts and run the PPO update in a torch-only subprocess
		(ppo_baseline/train_step.py). Returns avg training loss."""
		train_data = self._serialize_rollouts_for_training(updated_rollouts)
		print(f"   {len(train_data)} trainable episodes")

		next_iter = self.iteration + 1
		rollouts_pkl = self.checkpoint_dir / f"_rollouts_iter_{next_iter}.pkl"
		with open(rollouts_pkl, "wb") as f:
			pickle.dump(train_data, f)

		lora_out = str(self.checkpoint_dir / f"lora_iter_{next_iter}")
		opt_in = ""
		if self.current_lora_path and (Path(self.current_lora_path) / "optimizer.pt").exists():
			opt_in = str(Path(self.current_lora_path) / "optimizer.pt")
		metrics_out = str(Path(lora_out) / "train_metrics.json")

		cmd = [
			self.trainer_python, "-m", "ppo_baseline.train_step",
			"--base-model", self.config.base_model,
			"--rollouts", str(rollouts_pkl),
			"--lora-in", self.current_lora_path or "",
			"--lora-out", lora_out,
			"--optimizer-in", opt_in,
			"--optimizer-out", str(Path(lora_out) / "optimizer.pt"),
			"--metrics-out", metrics_out,
			"--epsilon", str(self.epsilon),
			"--lr", str(self.learning_rate),
			"--n-epochs", str(self.n_epochs),
			"--batch-size", str(self.batch_size),
			"--lora-r", str(self.lora_r),
			"--lora-alpha", str(self.lora_alpha),
			"--lora-dropout", str(self.lora_dropout),
			"--target-modules", self.lora_target_modules,
		]
		if self.load_in_4bit:
			cmd.append("--load-4bit")

		print(f"   Launching trainer: {self.trainer_python} -m ppo_baseline.train_step")
		subprocess.run(cmd, env=os.environ.copy(), check=True)

		# Trainer finished: advance iteration + adopt the new LoRA
		self.iteration = next_iter
		self.current_lora_path = lora_out
		print(f"💾 Trainer saved LoRA to {self.current_lora_path}")

		try:
			with open(metrics_out) as f:
				return json.load(f).get("avg_loss", 0.0)
		except Exception as e:
			print(f"   ⚠️  Could not read trainer metrics: {e}")
			return 0.0

	def train_iteration(self):
		"""Run one full training iteration with sequential model loading"""
		print(f"\n{'='*80}")
		print(f"PPO Iteration {self.iteration}")
		print(f"{'='*80}")
		
		# ================================
		# PHASE 1: ROLLOUTS WITH VLLM
		# ================================
		print("\n" + "="*80)
		print("PHASE 1: Collecting Rollouts with vLLM")
		print("="*80)
		
		# Start vLLM with current LoRA (or base model on iter 0)
		max_retries = 3
		for attempt in range(max_retries):
			if attempt > 0:
				print(f"\n🔄 Retry {attempt}/{max_retries-1}...")
			
			if self.start_vllm_server(lora_path=self.current_lora_path):
				break  # Success!
			
			if attempt < max_retries - 1:
				print(f"   Waiting 30s before retry...")
				time.sleep(30)
		else:
			raise RuntimeError(f"Failed to start vLLM after {max_retries} attempts")
		
		try:
			rollouts, task_set = self.collect_rollouts()
			updated_rollouts = self.get_advantages(rollouts, task_set)

			if self.use_memory:
				print("\n" + "="*80)
				print("Updating Memory Template")
				print("="*80)

				self.H = self._update_memory_template(updated_rollouts)

				# Save memory snapshot
				self.memory_history.append({
					'iteration': self.iteration,
					'H': self.H,
					'timestamp': time.time()
				})
				self._save_memory_snapshot()

		finally:
			self.stop_vllm_server()
			# Reduced sleep - 60s is too long
			print("   Waiting 10s for cleanup...")
			time.sleep(10)
		
		# ================================
		# PHASE 2: TRAINING IN A SEPARATE PROCESS (torch isolated from appworld)
		# ================================
		print("\n" + "="*80)
		print("PHASE 2: Training Policy Model (subprocess)")
		print("="*80)

		avg_loss = self._run_training_subprocess(updated_rollouts)

		# ================================
		# PHASE 3: SAVE CHECKPOINT
		# ================================
		print("\n" + "="*80)
		print("PHASE 3: Saving Checkpoint")
		print("="*80)

		# Compute metrics
		avg_reward = sum(r.get("overall_success", 0) or 0 for r in updated_rollouts) / len(updated_rollouts)
		successful_rollouts = sum(1 for r in updated_rollouts if r.get("completed", False))
		success_rate = (successful_rollouts / len(updated_rollouts)) * 100

		# Update training history
		self.training_history["iterations"].append(self.iteration)
		self.training_history["avg_rewards"].append(avg_reward)
		self.training_history["success_rates"].append(success_rate)
		self.training_history["avg_losses"].append(avg_loss)
		self.training_history["completed_tasks"].append(successful_rollouts)

		# Save checkpoint and metrics
		self.save_checkpoint()
		self.save_metrics()

		try:
			self.plot_training_curves()
		except Exception as e:
			print(f"⚠️  Failed to plot training curves: {e}")

		print(f"\n📊 Iteration {self.iteration} Summary:")
		print(f"   Average Reward: {avg_reward:.4f}")
		print(f"   Success Rate: {success_rate:.1f}%")
		print(f"   Average Loss: {avg_loss:.4f}")
		print(f"   Completed Tasks: {successful_rollouts}/{len(updated_rollouts)}")
		
		return updated_rollouts


def evaluate_lora(lora_path: str, dataset: str = "test_normal", max_tasks: int = None):
	"""
	Evaluate a trained LoRA on the test set.
	Uses the same evaluation logic as baseline/main.py
	"""
	from baseline.main import run_evaluation
	from baseline.config import Config
	
	print(f"\n{'='*80}")
	print(f"Evaluating LoRA: {lora_path}")
	print(f"Dataset: {dataset}")
	print(f"{'='*80}\n")
	
	# Create config with LoRA path
	config = Config()
	
	# Run evaluation (will pass lora_path to ReactAgent)
	experiment_name = f"ppo_eval_{Path(lora_path).name}"
	
	results = run_evaluation(
		dataset_name=dataset,
		experiment_name=experiment_name,
		max_tasks=max_tasks,
		config=config,
		lora_adapter_path=lora_path  # Pass LoRA to evaluation
	)
	
	return results


def main():
	import argparse
	# Set environment variables
	os.environ["OPENAI_API_KEY"] = "EMPTY"
	os.environ["NO_API_KEY"] = "EMPTY"
	os.environ["MODEL_SERVER_URL"] = "http://localhost:8000"
	parser = argparse.ArgumentParser(description="PPO-LOOP Training")
	parser.add_argument("--resume", type=str, default=None, 
					   help="Resume from checkpoint path")
	parser.add_argument("--eval-only", action="store_true",
					   help="Only evaluate a trained LoRA")
	parser.add_argument("--lora-path", type=str, default=None,
					   help="Path to LoRA for evaluation")
	parser.add_argument("--iterations", type=int, default=5,
					   help="Number of training iterations")
	parser.add_argument("--difficulties", type=int, nargs="+", default=[1, 2],
					   help="Task difficulty levels to train on (1, 2, and/or 3)")
	parser.add_argument("--checkpoint-dir", type=str, default=None,
					   help="Checkpoint dir. If omitted, defaults to "
							"./checkpoints/diff_<difficulties> so each difficulty "
							"trains a SEPARATE adapter (no overwrite).")
	# Rollout sampling (LOOP needs diverse trajectories -> nonzero advantage)
	parser.add_argument("--rollout-temperature", type=float, default=1.0,
					   help="Sampling temperature during rollouts (paper: 1.0)")
	parser.add_argument("--rollout-seed-base", type=int, default=1000,
					   help="Base seed; each rollout gets a unique seed offset")
	parser.add_argument("--train-max-iters", type=int, default=40,
					   help="Max agent<->env interactions per rollout (paper: 40 train, 50 eval)")
	parser.add_argument("--k", type=int, default=6, help="Rollouts per task (K)")
	parser.add_argument("--tasks-per-iter", type=int, default=40,
					   help="Tasks sampled per iteration (random_sample_number)")
	parser.add_argument("--n-epochs", type=int, default=1,
					   help="PPO inner epochs per iteration. 1 = 1-epoch token-LOOP "
							"(~3x faster than 3); paper's full LOOP uses >1.")
	parser.add_argument("--rollout-workers", type=int, default=1,
					   help="Parallel rollout processes (1 = sequential). ~max_num_seqs "
							"(e.g. 8) saturates the vLLM server. Cuts rollout wall-clock.")
	# vLLM serving (rollout phase)
	parser.add_argument("--tensor-parallel-size", type=int, default=2,
					   help="vLLM tensor-parallel size (GPUs for serving)")
	parser.add_argument("--gpu-mem-util", type=float, default=0.90,
					   help="vLLM --gpu-memory-utilization (vLLM runs alone in phase 1)")
	parser.add_argument("--max-model-len", type=int, default=20000,
					   help="vLLM --max-model-len")
	parser.add_argument("--max-num-seqs", type=int, default=8,
					   help="vLLM --max-num-seqs (rollouts are sequential; small fits 16GB)")
	parser.add_argument("--quantization", type=str, default=None,
					   help="vLLM --quantization (e.g. fp8); omit for none")
	# Training phase
	parser.add_argument("--load-4bit", action="store_true",
					   help="Load frozen base in 4-bit (QLoRA); requires bitsandbytes")
	# Portability
	parser.add_argument("--prompt-file", type=str, default=None,
					   help="Path to react_code_agent instructions.txt "
							"(default: $APPWORLD_PROMPT_FILE or local appworld repo)")
	# Prompt-memory (H) feature
	parser.add_argument("--use-memory", action="store_true",
					   help="Enable the Claude-generated prompt-memory (H) each "
							"iteration; requires ANTHROPIC_API_KEY. Off = pure RL.")
	# Trainer subprocess (torch isolated from appworld)
	parser.add_argument("--trainer-python", type=str, default=None,
					   help="Python with torch/transformers/peft for the training "
							"subprocess (default: $TRAINER_PYTHON or vllm_env python)")
	parser.add_argument("--vllm-bin", type=str, default=None,
					   help="Path to the vllm CLI (default: $VLLM_BIN or vllm_env/bin/vllm)")
	parser.add_argument("--lora-r", type=int, default=16)
	parser.add_argument("--lora-alpha", type=int, default=32)
	parser.add_argument("--lora-dropout", type=float, default=0.05)
	parser.add_argument("--target-modules", type=str,
					   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
					   help="Comma-separated LoRA target modules")

	args = parser.parse_args()

	# Per-difficulty checkpoint separation by default
	checkpoint_dir = args.checkpoint_dir
	if checkpoint_dir is None:
		diff_tag = "_".join(str(d) for d in args.difficulties)
		checkpoint_dir = f"./checkpoints/diff_{diff_tag}"
	
	# Evaluation mode
	if args.eval_only:
		if not args.lora_path:
			print("❌ Must provide --lora-path for evaluation")
			return
		evaluate_lora(args.lora_path, dataset="test_normal")
		return
	
	# Training mode
	config = Config()
	
	ppo_loop = PPO_LOOP(
		K=args.k,
		random_sample_number=args.tasks_per_iter,
		difficulties=args.difficulties,
		config=config,
		epsilon=0.2,
		learning_rate=5e-5,
		n_epochs=args.n_epochs,
		batch_size=3,
		checkpoint_dir=checkpoint_dir,
		resume_from=args.resume,
		rollout_temperature=args.rollout_temperature,
		rollout_seed_base=args.rollout_seed_base,
		train_max_iters=args.train_max_iters,
		rollout_workers=args.rollout_workers,
		vllm_tensor_parallel=args.tensor_parallel_size,
		vllm_gpu_mem_util=args.gpu_mem_util,
		vllm_max_model_len=args.max_model_len,
		vllm_max_num_seqs=args.max_num_seqs,
		vllm_quantization=args.quantization,
		load_in_4bit=args.load_4bit,
		prompt_file_path=args.prompt_file,
		use_memory=args.use_memory,
		lora_r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		lora_target_modules=args.target_modules,
		trainer_python=args.trainer_python,
		vllm_bin=args.vllm_bin,
	)

	start_iter = ppo_loop.iteration
	num_iterations = args.iterations
	
	print(f"\n🚀 Starting PPO-LOOP Training")
	print(f"   Base Model: {config.base_model}")
	print(f"   Iterations: {start_iter} → {start_iter + num_iterations}")
	print(f"   Difficulties: {args.difficulties}") 
	print(f"   Checkpoint dir: {ppo_loop.checkpoint_dir}")
	print(f"   vLLM Port: {ppo_loop.vllm_port}")
	
	try:
		for iteration in range(start_iter, start_iter + num_iterations):
			print(f"\n{'='*80}")
			print(f"Training Iteration {iteration+1}/{start_iter + num_iterations}")
			print(f"{'='*80}")
			
			rollouts = ppo_loop.train_iteration()
			
	except KeyboardInterrupt:
		print("\n⚠️  Training interrupted by user")
		print(f"   Last checkpoint saved at iteration {ppo_loop.iteration}")
	except Exception as e:
		print(f"\n❌ Training crashed: {e}")
		print(f"   Last checkpoint saved at iteration {ppo_loop.iteration}")
		import traceback
		traceback.print_exc()
		raise
	finally:
		# Cleanup: stop vLLM if still running
		print("\n🧹 Final cleanup...")
		ppo_loop.stop_vllm_server()
	
	print("\n✨ Training complete!")
	print(f"   Final LoRA: {ppo_loop.current_lora_path}")
	print(f"   Checkpoints: {ppo_loop.checkpoint_dir}")


if __name__ == "__main__":
	main()