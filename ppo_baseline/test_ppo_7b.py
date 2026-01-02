import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["NO_API_KEY"] = "EMPTY"
os.environ["MODEL_SERVER_URL"] = "http://localhost:8000"

from ppo_baseline.main_alt import PPO_LOOP
from shared.config import Config

# Configure for 7B model
config = Config()
config.base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
config.vllm_url = "http://localhost:8000/v1"
config.max_tokens = 1000
config.max_iters = 20
config.temperature = 0.0

# Create PPO trainer with SMALL settings for testing
ppo_loop = PPO_LOOP(
    K=2,                      # 2 rollouts per task
    random_sample_number=2,   # Only 2 tasks
    difficulties=[1],         # Easiest tasks only
    sets=["train"],           # Just training set
    config=config,
    epsilon=0.2,
    learning_rate=1e-5,
    n_epochs=2,
    batch_size=4,
    checkpoint_dir="./checkpoints_7b_test",
)

print("🧪 Running 3 test iterations with 7B model...")
print(f"   Model: {config.base_model}")
print(f"   Tasks per iter: {ppo_loop.random_sample_number}")
print(f"   Rollouts per task: {ppo_loop.K}")
print(f"   Total rollouts per iter: {ppo_loop.random_sample_number * ppo_loop.K}")

try:
    for i in range(3):
        print(f"\n{'='*80}")
        print(f"Test Iteration {i+1}/3")
        print(f"{'='*80}")
        
        rollouts = ppo_loop.train_iteration()
        
        # Check results
        avg_reward = sum(r["overall_success"] for r in rollouts) / len(rollouts)
        successful = sum(1 for r in rollouts if r["completed"])
        
        print(f"\n📊 Iteration {i+1} Results:")
        print(f"   Avg Reward: {avg_reward:.3f}")
        print(f"   Success Rate: {successful}/{len(rollouts)}")
        print(f"   Reward History: {ppo_loop.training_history['avg_rewards']}")
        
        # Verify logprobs
        sample_rollout = rollouts[0]
        agent_state = sample_rollout["agent_state"]
        assistant_msgs = [m for m in agent_state.conversation_history if m.role == "assistant"]
        msgs_with_logprobs = [m for m in assistant_msgs if m.log_probs]
        
        print(f"   Logprobs check: {len(msgs_with_logprobs)}/{len(assistant_msgs)} messages have logprobs")
        
        if len(msgs_with_logprobs) == 0:
            print("   ❌ WARNING: No logprobs found!")
            break

except KeyboardInterrupt:
    print("\n⚠️  Test interrupted")
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete!")
print(f"   Checkpoints saved to: {ppo_loop.checkpoint_dir}")