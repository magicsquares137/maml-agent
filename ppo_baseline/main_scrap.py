import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import uuid
from appworld import AppWorld, load_task_ids


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
        batch_size: int = 8
    ):
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
            print(f"Init difficulty check task {i} of {len(train_ids)}")
            temp_world = AppWorld(task_id=x)
            if world.task.ground_truth.metadata["difficulty"] in difficulties:
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
        
        # Start with None - first rollouts use base model
        self.current_lora_path = None
        
        # Will initialize after first rollout collection
        self.policy_model = None
        self.optimizer = None
    
    def collect_rollouts(
        self
    ) -> List[dict]:

        # Collect task ids
        task_set = random.sample(self.train_ids, self.random_sample_number)
        all_rollouts = []

        for index, task_id in enumerate(
            tqdm(
                task_set, 
                desc=f" Running base policy rollouts for {len(task_set)} tasks"
                )
            ):
            print(f"\n{'='*60}")
            print(f"Task {index + 1}/{len(task_set)}: {task_id}")
            print(f"{'='*60}")

            for rollout in range(self.K):
                print(f"\n{'='*60}")
                print(f"Task {task_id} rollout: {rollout}")
                print(f"{'='*60}")

                # Create fresh agent for each rollout
                agent = ReactAgent(
                    self.config, 
                    lora_adapter_path=self.current_lora_path,
                )
                random_uuid = uuid.uuid4()   

                # Note: each dict below will end up being around .45KB            
                task_result = {
                    "task_id": task_id,
                    "completed": False,
                    "iterations": 0,
                    "error": None,
                    "conversation_length": 0,
                    "overall_success": None,
                    "uuid": random_uuid,
                    "agent_state": None,
                    "evaluation_details": None
                }

                try:
                    # Load the appworld environment for the task
                    with AppWorld(
                        task_id=task_id,
                        experiment_name="ppo_training",
                    ) as world: 
                        print(f"📋 Instruction: {world.task.instruction}\n")
                        agent.initialize(
                            first_name=world.task.supervisor.get("first_name", ""),
                            last_name=world.task.supervisor.get("last_name", ""),
                            email=world.task.supervisor.get("email", ""),
                            phone_number=world.task.supervisor.get("phone_number", ""),
                            task_instructions=world.task.instruction
                        )   

                        # Complete the task
                        agent.run(world) 

                        # Collect results
                        task_result["completed"] = world.task_completed()
                        task_result["iterations"] = agent.state.iteration
                        task_result["conversation_length"] = len(agent.state.conversation_history)
                            
                        # Get performance metrics
                        evaluation = world.evaluate().to_dict()
                        
                        task_result["overall_success"] = len(evaluation['passes'])/evaluation['num_tests']
                        task_result["evaluation_details"] = evaluation
                    
                    print(f"\nTask finished: {task_result['completed']}")
                    print(f"🔄 Iterations: {task_result['iterations']}")
                except Exception as e:
                    task_result["error"] = str(e)
                    print(f"\n Error: {e}")

                # Attach the agents state to cache log probs/tokens
                task_result["agent_state"] = agent.state

                all_rollouts.append(task_result)

        return all_rollouts, task_set

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
                    f"Expected {self.K} rollouts for task {task}, but found {actual_rollouts}. "
                    f"Check if collect_rollouts() completed successfully."
                )
            
            # Compute average reward across ALL K rollouts
            avg_reward = sum(x["overall_success"] for x in task_rollouts) / self.K
            
            for rollout in task_rollouts:
                rollout_reward = rollout["overall_success"]
                
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
            with torch.no_grad():
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
                new_logprob = log_probs[position, token_id].item()
                
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
        total_loss = 0.0
        num_tokens = 0
        
        for episode in minibatch:
            # Compute new log probs and get token data
            episode = self.compute_log_probs(episode)  
            token_data = episode["token_level_data"] 
            
            advantage = episode["advantage"]
            
            for token_info in token_data:
                new_logprob = token_info['new_logprob']
                old_logprob = token_info['old_logprob']
                
                # Importance ratio: π_new(token) / π_old(token)
                log_ratio = new_logprob - old_logprob
                ratio = torch.exp(torch.tensor(log_ratio, dtype=torch.float32))
                
                # Standard PPO clipping: min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                
                surrogate1 = ratio * advantage
                surrogate2 = clipped_ratio * advantage
                
                # Take minimum and negate (we want to maximize, optimizer minimizes)
                token_loss = -torch.min(surrogate1, surrogate2)
                
                total_loss += token_loss
                num_tokens += 1
        
        # Average over all tokens in minibatch
        return total_loss / num_tokens if num_tokens > 0 else torch.tensor(0.0, dtype=torch.float32)
    
    def shuffled_batchify(self, data, batch_size):
        indices = list(range(len(data)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            yield [data[j] for j in batch_idx]
    
    def train_iteration(self):
        """Run one full training iteration"""
        rollouts, task_set = self.collect_rollouts()
        
        # 2. Compute advantages
        updated_rollouts = self.get_advantages(rollouts, task_set)
        
        # 3. Update policy with PPO
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for minibatch in self.shuffled_batchify(updated_rollouts, self.batch_size):
                self.optimizer.zero_grad()
                
                loss = self.compute_ppo_loss(minibatch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            print(f"Epoch {epoch+1}/{self.n_epochs}, Avg Loss: {epoch_loss/num_batches:.4f}")
        
        return updated_rollouts

def main():
    config = Config()  # Your config
    
    ppo_loop = PPO_LOOP(
        K=6,
        random_sample_number=40,
        config=config,
        epsilon=0.2,
        learning_rate=1e-5,
        n_epochs=3,
        batch_size=8
    )
    
    num_iterations = 10
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration+1}/{num_iterations}")
        print(f"{'='*60}")
        
        rollouts = ppo_loop.train_iteration()
        
        # Save checkpoint
        ppo_loop.policy_model.save_pretrained(f"checkpoints/iter_{iteration}")
        
        # Log metrics
        avg_reward = sum(r["overall_success"] for r in rollouts) / len(rollouts)
        print(f"Average Reward: {avg_reward:.4f}")

if __name__ == "__main__":
    main()
