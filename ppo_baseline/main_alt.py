import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import uuid
from appworld import AppWorld, load_task_ids
from shared.config import Config
from shared.agent import ReactAgent
from shared.models import AgentState, Message
from typing import Dict, Optional, List, Union, Dict
from appworld import AppWorld, load_task_ids
from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent
from pathlib import Path
from tqdm import tqdm
import subprocess
import time
import signal
import os 
import requests
import time


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
		resume_from: str = None
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
		
		# Initialize LoRA config (but don't apply it yet)
		self.lora_config = LoraConfig(
			r=16,
			lora_alpha=32,
			target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
			lora_dropout=0.05,
			bias="none",
			task_type="CAUSAL_LM"
		)

		self.vllm_process = None
		self.vllm_port = 8000
		self.vllm_host = "localhost"

	def start_vllm_server(self, lora_path: str | None = None) -> bool:
		print("\n🚀 Starting vLLM server...")

		cmd = [
			"vllm", "serve", self.config.base_model,
			"--host", self.vllm_host,                 # important if not localhost
			"--port", str(self.vllm_port),
			"--max-model-len", "25192",
			"--gpu-memory-utilization", "0.45",
			"--enable-lora",
			"--max-loras", "2",
			"--max-lora-rank", "64",
		]

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

		max_wait_time = 360
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
			
			# Nuclear option: kill ALL vLLM and Ray processes
			print("   Cleaning up vLLM/Ray processes...")
			subprocess.run(["pkill", "-9", "-f", "vllm"], stderr=subprocess.DEVNULL)
			subprocess.run(["pkill", "-9", "-f", "ray::"], stderr=subprocess.DEVNULL)
			subprocess.run(["pkill", "-9", "-f", "_raylet"], stderr=subprocess.DEVNULL)
			
			# Also kill Ray completely
			try:
				import ray
				if ray.is_initialized():
					ray.shutdown()
			except:
				pass
			
			# Force Ray shutdown via CLI
			subprocess.run(["ray", "stop", "--force"], 
						  stderr=subprocess.DEVNULL, 
						  stdout=subprocess.DEVNULL)
			
			# Close log files if they exist
			if hasattr(self, 'vllm_stdout_file'):
				self.vllm_stdout_file.close()
			if hasattr(self, 'vllm_stderr_file'):
				self.vllm_stderr_file.close()
			
			# Wait for GPU memory to actually be freed
			print("   Waiting for GPU cleanup...", end="", flush=True)
			time.sleep(10)  # Increased from 5 to 10 seconds
			
			# Force CUDA cache clear
			torch.cuda.empty_cache()
			
			# Verify memory is freed
			if torch.cuda.is_available():
				torch.cuda.synchronize()
				allocated = torch.cuda.memory_allocated() / 1024**3
				reserved = torch.cuda.memory_reserved() / 1024**3
				print(f" Done")
				print(f"   GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
			else:
				print(" Done")

	def save_checkpoint(self):
		"""Save full training checkpoint"""
		checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
		
		checkpoint = {
			"iteration": self.iteration,
			"current_lora_path": self.current_lora_path,
			"training_history": self.training_history,
			"optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
			"config": {
				"K": self.K,
				"random_sample_number": self.random_sample_number,
				"epsilon": self.epsilon,
				"learning_rate": self.learning_rate,
				"n_epochs": self.n_epochs,
				"batch_size": self.batch_size
			}
		}
		
		torch.save(checkpoint, checkpoint_path)
		print(f"💾 Saved checkpoint to {checkpoint_path}")
		
		# Also save a "latest" checkpoint
		latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
		torch.save(checkpoint, latest_path)

	def _initialize_policy_model(self):
		"""Initialize policy model for training"""
		print("\n🏋️ Initializing policy model for training...")
		
		base_model = AutoModelForCausalLM.from_pretrained(
			self.config.base_model,
			torch_dtype=torch.float16,
			device_map="auto",
			trust_remote_code=True  # For Qwen models
		)
		base_model.gradient_checkpointing_enable()
		
		# Freeze base model
		for param in base_model.parameters():
			param.requires_grad = False
		
		# Apply LoRA
		policy_model = get_peft_model(base_model, self.lora_config)
		
		# Load existing LoRA weights if available
		if self.current_lora_path and Path(self.current_lora_path).exists():
			print(f"   Loading existing LoRA: {self.current_lora_path}")
			# Load the adapter weights
			adapter_weights = torch.load(
				Path(self.current_lora_path) / "adapter_model.bin"
			)
			policy_model.load_state_dict(adapter_weights, strict=False)
		
		# Initialize optimizer
		self.optimizer = torch.optim.AdamW(
			policy_model.parameters(),
			lr=self.learning_rate
		)
		
		print("   Policy model ready")
		return policy_model

	def _cleanup_policy_model(self):
		"""Unload policy model and free GPU memory"""
		print("\n🧹 Cleaning up policy model...")
		
		if hasattr(self, 'policy_model') and self.policy_model is not None:
			del self.policy_model
			self.policy_model = None
		
		if hasattr(self, 'optimizer') and self.optimizer is not None:
			del self.optimizer
			self.optimizer = None
		
		import gc
		gc.collect()
		torch.cuda.empty_cache()
		
		# Verify GPU memory freed
		if torch.cuda.is_available():
			allocated = torch.cuda.memory_allocated() / 1024**3
			print(f"   GPU Memory Allocated: {allocated:.2f} GB")
		
		print("   Policy model unloaded")

	def load_checkpoint(self, checkpoint_path: str):
		"""Load training checkpoint to resume"""
		print(f"📂 Loading checkpoint from {checkpoint_path}")
		
		checkpoint = torch.load(checkpoint_path)
		
		self.iteration = checkpoint["iteration"]
		self.current_lora_path = checkpoint["current_lora_path"]
		self.training_history = checkpoint["training_history"]
		
		print(f"✅ Resumed from iteration {self.iteration}")
		print(f"   Current LoRA: {self.current_lora_path}")
	
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
		all_rollouts = []
		
		for index, task_id in enumerate(tqdm(task_set, desc=f"Running rollouts")):
			print(f"\n{'='*60}")
			print(f"Task {index + 1}/{len(task_set)}: {task_id}")
			print(f"{'='*60}")
			
			for rollout in range(self.K):
				print(f"\nRollout {rollout}")

				extra_body = {
					"return_token_ids": True
				}
				
				# If using LoRA, specify which one
				if self.current_lora_path:
					extra_body["lora_name"] = "ppo_adapter"

				agent = SimplifiedReActCodeAgent(
					model_config={
						"client_name": "openai",
						"api_type": "chat_completions",
						"base_url": self.config.vllm_url,
						"name": self.config.base_model,
						"api_key_env_name": "NO_API_KEY",
						"temperature": self.config.temperature,
						"seed": 100,
						"logprobs": True,
						"top_logprobs": 1,
						"extra_body": extra_body,
						"max_completion_tokens": self.config.max_tokens,
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
					logger_config={"color": True, "verbose": False},
					appworld_config={"random_seed": 100},
					prompt_file_path="/workspace/appworld/appworld/experiments/prompts/react_code_agent/instructions.txt",
					ignore_multiple_calls=True,
					max_prompt_length=None,
					max_output_length=None,
					max_steps=self.config.max_iters,
				)
				
				task_result = {
					"task_id": task_id,
					"completed": False,
					"iterations": 0,
					"error": None,
					"conversation_length": 0,
					"overall_success": None,
					"uuid": uuid.uuid4(),
					"agent_state": None,
					"evaluation_details": None
				}
				
				try:  # INDENT THIS - INSIDE THE LOOP!
					agent.logger.initialize("ppo_training", len(task_set) * self.K, 1, 0)
					agent.solve_task(task_id)
					
					#completed = agent.world.task_completed()
					evaluation = agent.world.evaluate().to_dict()
					completed = evaluation["success"]
					overall_success = len(evaluation['passes']) / evaluation['num_tests']
					
					agent_state = self.convert_to_agent_state(agent)
					agent_state.done = completed
					
					task_result["completed"] = completed
					task_result["iterations"] = agent.step_number
					task_result["conversation_length"] = len(agent_state.conversation_history)
					task_result["overall_success"] = overall_success
					task_result["evaluation_details"] = evaluation
					task_result["agent_state"] = agent_state
					
					print(f"✅ Success: {overall_success:.3f}")
					
				except Exception as e:
					task_result["error"] = str(e)
					print(f"❌ Error: {e}")
					task_result["agent_state"] = None
				
				finally:
					# Clean up database
					# if hasattr(agent, 'world'):
					# 	agent.world.close()
					del agent
				
				all_rollouts.append(task_result)
		
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

	def compute_log_probs(self, episode: dict) -> dict:
		"""
		Compute NEW log probabilities for tokens generated during rollout.
		
		Uses exact token IDs from vLLM (stored during rollout) to avoid
		retokenization drift. Computes P_new(tokens) where tokens were
		generated by the old policy.
		
		Args:
			episode: Task result dict with agent_state containing:
				- conversation_history: List[Message] with assistant messages
				- Each assistant message has:
					- tokenized_input: input token IDs from vLLM
					- log_probs: [(token_str, old_logprob, token_id), ...]
		
		Returns:
			episode dict with added "token_level_data" field containing:
			[
				{
					'new_logprob': float,  # P_new(token | context)
					'old_logprob': float,  # P_old(token | context) from rollout
					'token_id': int,
					'token_str': str,
					'assistant_turn': int,  # Which assistant message
					'position': int,  # Position within that message
				},
				...
			]
		"""
		all_token_data = []
		agent_state = episode["agent_state"]
		assistant_idx = 0
		
		for i, msg in enumerate(agent_state.conversation_history):
			if msg.role != "assistant":
				continue
			
			# Skip if no log probs (shouldn't happen but defensive)
			if msg.log_probs is None or msg.tokenized_input is None:
				continue
			
			# Get exact token IDs from rollout (no retokenization!)
			prompt_token_ids = msg.tokenized_input  # Input tokens
			output_token_data = msg.log_probs  # [(token_str, old_logprob, token_id), ...]
			output_token_ids = [token_id for _, _, token_id in output_token_data]
			
			# Concatenate: full sequence = prompt + output
			full_token_ids = prompt_token_ids + output_token_ids
			full_ids = torch.tensor([full_token_ids], device=self.policy_model.device)  # [1, seq_len]
			
			# Forward pass through NEW policy
			outputs = self.policy_model(full_ids)
			logits = outputs.logits[0]  # [seq_len, vocab_size]
			
			# Compute log probabilities
			log_probs = torch.log_softmax(logits, dim=-1)  # [seq_len, vocab_size]
			
			# For each output token, get its NEW log prob
			prompt_length = len(prompt_token_ids)
			
			for j, (token_str, old_logprob, token_id) in enumerate(output_token_data):
				# KEY: Autoregressive shift
				# Token at position prompt_length + j is predicted by logits at position prompt_length + j - 1
				# Because: logits[t] predicts token[t+1]
				position = prompt_length + j - 1
				
				# Sanity check
				if position < 0 or position >= log_probs.shape[0]:
					print(f"Warning: position {position} out of range for sequence length {log_probs.shape[0]}")
					continue
				
				# Get NEW policy's log probability for this exact token
				new_logprob = log_probs[position, token_id]
				
				all_token_data.append({
					'new_logprob': new_logprob,
					'old_logprob': old_logprob,
					'token_id': token_id,
					'token_str': token_str,
					'assistant_turn': assistant_idx,
					'position': j,
				})
			
			assistant_idx += 1
		
		# Add token data to episode
		episode["token_level_data"] = all_token_data
		
		return episode

	def compute_ppo_loss(self, minibatch):
		"""
		Compute PPO loss for a minibatch of episodes.
		Uses per-token importance weights (Equation 5 from paper).
		"""
		total_loss = torch.tensor(0.0, device=self.policy_model.device)
		num_tokens = 0
		
		for episode in minibatch:
			# Skip episodes that failed or have no agent_state
			if episode.get("agent_state") is None:
				print(f"⚠️  Skipping episode {episode.get('task_id')} - no agent_state")
				continue
				
			# Skip episodes with errors
			if episode.get("error") is not None:
				print(f"⚠️  Skipping episode {episode.get('task_id')} - error: {episode.get('error')}")
				continue
			
			# Compute new log probs and get token data
			episode = self.compute_log_probs(episode)  
			token_data = episode.get("token_level_data", [])
			
			# Skip if no tokens
			if len(token_data) == 0:
				print(f"⚠️  Skipping episode {episode.get('task_id')} - no tokens")
				continue
			
			advantage = episode["advantage"]
			
			for token_info in token_data:
				new_logprob = token_info['new_logprob']
				old_logprob = token_info['old_logprob']
				
				# Importance ratio: π_new(token) / π_old(token)
				log_ratio = new_logprob - old_logprob
				ratio = torch.exp(log_ratio)
				
				# Standard PPO clipping: min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
				clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
				
				surrogate1 = ratio * advantage
				surrogate2 = clipped_ratio * advantage
				
				# Take minimum and negate (we want to maximize, optimizer minimizes)
				token_loss = -torch.min(surrogate1, surrogate2)
				total_loss = total_loss + token_loss
				num_tokens += 1
		
		# Average over all tokens in minibatch
		if num_tokens == 0:
			print("⚠️  WARNING: No valid tokens in minibatch! Returning zero loss.")
			return torch.tensor(0.0, device=self.policy_model.device, requires_grad=True)
		
		return total_loss / num_tokens
	
	def shuffled_batchify(self, data, batch_size):
		indices = list(range(len(data)))
		random.shuffle(indices)
		
		for i in range(0, len(indices), batch_size):
			batch_idx = indices[i:i + batch_size]
			yield [data[j] for j in batch_idx]
	
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
		if not self.start_vllm_server(lora_path=self.current_lora_path):
			raise RuntimeError("Failed to start vLLM server!")
		
		try:
			# Collect rollouts
			rollouts, task_set = self.collect_rollouts()
			updated_rollouts = self.get_advantages(rollouts, task_set)
		finally:
			# Always stop vLLM, even if rollouts fail
			self.stop_vllm_server()
			time.sleep(60)
		
		# ================================
		# PHASE 2: TRAINING WITH POLICY MODEL
		# ================================
		print("\n" + "="*80)
		print("PHASE 2: Training Policy Model")
		print("="*80)
		
		# Initialize policy model (now that vLLM is stopped)
		if self.policy_model is None:
			self.policy_model = self._initialize_policy_model()
		
		# PPO training loop
		total_epoch_loss = 0
		for epoch in range(self.n_epochs):
			epoch_loss = 0
			num_batches = 0
			
			for minibatch in self.shuffled_batchify(updated_rollouts, self.batch_size):
				self.optimizer.zero_grad()
				
				loss = self.compute_ppo_loss(minibatch)
				
				if loss.requires_grad:
					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
					self.optimizer.step()
				
				epoch_loss += loss.item()
				num_batches += 1
			
			avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
			total_epoch_loss += avg_epoch_loss
			print(f"   Epoch {epoch+1}/{self.n_epochs}, Avg Loss: {avg_epoch_loss:.4f}")
		
		# ================================
		# PHASE 3: SAVE AND CLEANUP
		# ================================
		print("\n" + "="*80)
		print("PHASE 3: Saving Checkpoint and Cleaning Up")
		print("="*80)
		
		# Save updated LoRA
		self.iteration += 1
		self.current_lora_path = str(self.checkpoint_dir / f"lora_iter_{self.iteration}")
		self.policy_model.save_pretrained(self.current_lora_path)
		print(f"💾 Saved LoRA to {self.current_lora_path}")
		
		# Compute metrics
		avg_reward = sum(r.get("overall_success", 0) or 0 for r in updated_rollouts) / len(updated_rollouts)
		successful_rollouts = sum(1 for r in updated_rollouts if r.get("completed", False))
		success_rate = (successful_rollouts / len(updated_rollouts)) * 100
		avg_loss = total_epoch_loss / self.n_epochs if self.n_epochs > 0 else 0
		
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
		
		# Clean up policy model to free GPU memory for next iteration
		self._cleanup_policy_model()
		
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
	parser.add_argument("--iterations", type=int, default=10,
					   help="Number of training iterations")
	
	args = parser.parse_args()
	
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
		K=2,
		random_sample_number=2,
		config=config,
		epsilon=0.2,
		learning_rate=1e-5,
		n_epochs=2,
		batch_size=2,
		checkpoint_dir="./checkpoints",
		resume_from=args.resume
	)
	
	start_iter = ppo_loop.iteration
	num_iterations = args.iterations
	
	print(f"\n🚀 Starting PPO-LOOP Training")
	print(f"   Base Model: {config.base_model}")
	print(f"   Iterations: {start_iter} → {start_iter + num_iterations}")
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
		ppo_loop._cleanup_policy_model()
	
	print("\n✨ Training complete!")
	print(f"   Final LoRA: {ppo_loop.current_lora_path}")
	print(f"   Checkpoints: {ppo_loop.checkpoint_dir}")


if __name__ == "__main__":
	main()