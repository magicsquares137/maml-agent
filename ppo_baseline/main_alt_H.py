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
import json
import matplotlib.pyplot as plt


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
		
		# Initialize LoRA config (but don't apply it yet)
		self.lora_config = LoraConfig(
			r=16,
			lora_alpha=32,
			target_modules=[
				# Self-attention modules <- per paper
				"q_proj", "k_proj", "v_proj", "o_proj",
				# MLP modules
				"gate_proj", "up_proj", "down_proj"
			],
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
			"--max-model-len", "30000",
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
			torch_dtype=torch.bfloat16,
			device_map="auto",
			trust_remote_code=True
		)
		base_model.gradient_checkpointing_enable()
		
		# Freeze base model
		for param in base_model.parameters():
			param.requires_grad = False
		
		# Check if we should load existing LoRA or create fresh one
		if self.current_lora_path and Path(self.current_lora_path).exists():
			print(f"   Loading existing LoRA from: {self.current_lora_path}")
			try:
				# Use PEFT's proper loading method
				policy_model = PeftModel.from_pretrained(
					base_model,
					self.current_lora_path,
					is_trainable=True  # Important: make it trainable
				)
				print(f"   ✅ Loaded existing LoRA")
			except Exception as e:
				print(f"   ⚠️  Failed to load LoRA: {e}")
				print(f"   Creating fresh LoRA instead")
				policy_model = get_peft_model(base_model, self.lora_config)
		else:
			print("   Creating fresh LoRA")
			policy_model = get_peft_model(base_model, self.lora_config)
		
		# Initialize optimizer
		self.optimizer = torch.optim.AdamW(
			policy_model.parameters(),
			lr=self.learning_rate
		)
		
		print("   ✅ Policy model ready")
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
	    Create temporary prompt file with memory template prepended
	    """
	    # NOTE: may want to add this somewhere like first message instead? or inject before final job instructions?
	    original_prompt_path = "/workspace/appworld/appworld/experiments/prompts/react_code_agent/instructions.txt"
	    
	    # If no memory yet (iteration 0), use original
	    if not self.H or len(self.H.strip()) == 0:
	        print("   Using original prompt (no memory yet)")
	        return original_prompt_path
	    
	    # Read original prompt
	    with open(original_prompt_path, 'r') as f:
	        original_prompt = f.read()
	    
	    # Prepend memory template
	    modified_prompt = f"""### BEST PRACTICES (learned from previous iterations):

	{self.H}

	### END BEST PRACTICES
	---

	{original_prompt}
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
		all_rollouts = []

	    # Create prompt with memory injected
	    temp_prompt_path = self._create_prompt_with_memory()
		
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
					prompt_file_path=temp_prompt_path,
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

	def _update_memory_template(self, rollouts: List[dict]) -> str:
	    """
	    Generate best practices template by feeding all conversation history to Claude
	    """
	    print("\n📝 Generating best practices template from rollouts...")
	    
	    # Build the input for Claude
	    conversations_text = self._format_rollouts_for_llm(rollouts)
	    
	    # Call Claude to synthesize best practices
	    prompt = f"""You are analyzing conversation logs from an AI agent attempting to solve AppWorld tasks.

	Here are {len(rollouts)} task execution logs from iteration {self.iteration}:

	{conversations_text}

	Previous best practices template from iteration {self.iteration - 1}:
	{self.H if self.H else "[None - this is the first iteration]"}

	Your task: Create a concise best practices template (max 600 words) that will be injected into the agent's system prompt to improve future performance.

	Focus on:
	1. Common failure patterns and how to avoid them (authentication, loops, giving up early, etc.)
	2. Successful strategies that worked
	3. Task-specific guidance (e.g., "for shopping tasks, always X before Y")

	Write the template in a clear, actionable format that the agent can follow.

	Best Practices Template:
	"""
	    
	    new_H = self._call_claude_api(prompt)
	    
	    print(f"   ✅ Generated template ({len(new_H)} characters)")
	    return new_H

	def _format_rollouts_for_llm(self, rollouts: List[dict]) -> str:
	    """
	    Format rollout conversations into text for LLM analysis
	    """
	    formatted = []
	    
	    # Take all rollouts (or sample if too many)
	    sample_size = min(len(rollouts), 40)  # Claude can handle this
	    sampled = rollouts[:sample_size]
	    
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
	            f"Rollout {i+1}/{sample_size}: {success}",
	            f"Task: {task_id}",
	            f"Steps: {steps}",
	            f"Score: {score:.2f}",
	            f"{'='*60}\n"
	        ]
	        
	        # Add conversation history
	        for msg in agent_state.conversation_history:
	            # Format: "ROLE: content"
	            rollout_text.append(f"{msg.role.upper()}: {msg.content}\n")
	        
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
	            model="claude-sonnet-4-5",
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
	        return self.H 	

	def compute_log_probs(self, episode: dict) -> dict:
		"""
		Compute NEW log probabilities for tokens generated during rollout.
		NOTE: This runs in NO_GRAD mode to save memory - gradients computed later in loss.
		"""
		all_token_data = []
		agent_state = episode["agent_state"]
		assistant_idx = 0
		
		# NO GRADIENTS during log prob computation
		with torch.no_grad():
			for i, msg in enumerate(agent_state.conversation_history):
				if msg.role != "assistant":
					continue
				
				if msg.log_probs is None or msg.tokenized_input is None:
					continue
				
				prompt_token_ids = msg.tokenized_input
				output_token_data = msg.log_probs
				output_token_ids = [token_id for _, _, token_id in output_token_data]
				
				full_token_ids = prompt_token_ids + output_token_ids
				full_ids = torch.tensor([full_token_ids], device=self.policy_model.device)
				
				# Forward pass WITHOUT gradients
				outputs = self.policy_model(full_ids)
				logits = outputs.logits[0]
				log_probs = torch.log_softmax(logits, dim=-1)
				
				prompt_length = len(prompt_token_ids)
				
				for j, (token_str, old_logprob, token_id) in enumerate(output_token_data):
					position = prompt_length + j - 1
					
					if position < 0 or position >= log_probs.shape[0]:
						continue
					
					# Extract as Python float (detaches from graph)
					new_logprob = log_probs[position, token_id].item()
					
					all_token_data.append({
						'new_logprob': new_logprob,  # Now a float, not a tensor
						'old_logprob': old_logprob,
						'token_id': token_id,
						'token_str': token_str,
						'assistant_turn': assistant_idx,
						'position': j,
					})
				
				assistant_idx += 1
				
				# Clear CUDA cache after each message
				del outputs, logits, log_probs, full_ids
				torch.cuda.empty_cache()
		
		episode["token_level_data"] = all_token_data
		return episode

	def compute_ppo_loss(self, minibatch):
		"""
		Compute PPO loss for a minibatch of episodes.
		Recomputes forward passes with gradients for the actual loss.
		"""
		total_loss = torch.tensor(0.0, device=self.policy_model.device, requires_grad=True)
		num_tokens = 0
		
		for episode in minibatch:
			if episode.get("agent_state") is None or episode.get("error") is not None:
				continue
			
			# Get precomputed log probs (no gradients)
			episode = self.compute_log_probs(episode)  
			token_data = episode.get("token_level_data", [])
			
			if len(token_data) == 0:
				continue
			
			advantage = torch.tensor(episode["advantage"], device=self.policy_model.device)
			
			# Process tokens in smaller chunks to save memory
			for token_info in token_data:
				new_logprob = torch.tensor(
					token_info['new_logprob'], 
					device=self.policy_model.device,
					requires_grad=False  # This is just data
				)
				old_logprob = torch.tensor(
					token_info['old_logprob'],
					device=self.policy_model.device,
					requires_grad=False
				)
				
				# Importance ratio
				log_ratio = new_logprob - old_logprob
				ratio = torch.exp(log_ratio)
				
				# PPO clipping
				clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
				
				surrogate1 = ratio * advantage
				surrogate2 = clipped_ratio * advantage
				
				token_loss = -torch.min(surrogate1, surrogate2)
				total_loss = total_loss + token_loss
				num_tokens += 1
		
		if num_tokens == 0:
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
		# PHASE 2: TRAINING WITH POLICY MODEL
		# ================================
		print("\n" + "="*80)
		print("PHASE 2: Training Policy Model")
		print("="*80)
		
		# Initialize policy model (now that vLLM is stopped)
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
		K=6,
		random_sample_number=40,
		config=config,
		epsilon=0.2,
		learning_rate=5e-5,
		n_epochs=3,
		batch_size=3,
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