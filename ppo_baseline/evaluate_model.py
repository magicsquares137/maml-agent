# evaluate_model.py
import argparse
import subprocess
import time
import signal
import os
import requests
from pathlib import Path
from appworld import load_task_ids
from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent

def start_vllm(model_path: str, port: int = 8000):
    """Start vLLM server"""
    cmd = [
        "vllm", "serve", model_path,
        "--host", "localhost",
        "--port", str(port),
        "--max-model-len", "30000",
        "--gpu-memory-utilization", "0.45",
    ]
    
    log_file = open("vllm_eval.log", "ab")
    process = subprocess.Popen(
        cmd, 
        stdout=log_file, 
        stderr=log_file,
        start_new_session=True
    )
    
    # Wait for ready
    print(f"   Starting vLLM (PID {process.pid})...")
    for i in range(300):
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=1)
            if r.status_code == 200:
                print(f"   ✅ vLLM ready!")
                return process
        except:
            pass
        time.sleep(1)
        if i % 10 == 0 and i > 0:
            print(".", end="", flush=True)
    
    raise RuntimeError("vLLM failed to start")

def stop_vllm(process):
    """Stop vLLM server"""
    print(f"\n   Stopping vLLM...")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        time.sleep(3)
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except:
        pass
    
    # Nuclear cleanup
    subprocess.run(["pkill", "-9", "-f", "vllm"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "ray"], stderr=subprocess.DEVNULL)
    subprocess.run(["ray", "stop", "--force"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    
    time.sleep(10)

def create_prompt_with_memory(memory_path: str | None) -> str:
    """Create prompt file with memory template"""
    original_prompt = "/workspace/appworld/appworld/experiments/prompts/react_code_agent/instructions.txt"
    
    if not memory_path:
        return original_prompt
    
    # Read memory
    with open(memory_path, 'r') as f:
        memory = f.read()
    
    # Read original
    with open(original_prompt, 'r') as f:
        original = f.read()
    
    # Combine
    combined = f"""### BEST PRACTICES:

{memory}

### END BEST PRACTICES
---

{original}
"""
    
    # Save temp
    temp_path = "./temp_prompt_with_memory.txt"
    with open(temp_path, 'w') as f:
        f.write(combined)
    
    return temp_path

def evaluate(model_path: str, memory_path: str | None, dataset: str, max_tasks: int = None):
    """Evaluate model on dataset"""
    print(f"\n{'='*80}")
    print(f"EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Memory: {memory_path or 'None'}")
    print(f"Dataset: {dataset}")
    print(f"{'='*80}\n")
    
    # Start vLLM
    vllm_process = start_vllm(model_path)
    
    try:
        # Create prompt
        prompt_path = create_prompt_with_memory(memory_path)
        
        # Load tasks
        task_ids = load_task_ids(dataset)
        if max_tasks:
            task_ids = task_ids[:max_tasks]
        
        print(f"\n   Evaluating {len(task_ids)} tasks...")
        
        results = []
        for i, task_id in enumerate(task_ids):
            print(f"\n   Task {i+1}/{len(task_ids)}: {task_id}")
            
            agent = SimplifiedReActCodeAgent(
                model_config={
                    "client_name": "openai",
                    "api_type": "chat_completions",
                    "base_url": "http://localhost:8000/v1",
                    "name": model_path,
                    "api_key_env_name": "NO_API_KEY",
                    "temperature": 0.7,
                    "seed": 100,
                    "max_completion_tokens": 2048,
                    "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "output": 0.0},
                },
                logger_config={"color": True, "verbose": False},
                prompt_file_path=prompt_path,
                max_steps=15,
            )
            
            try:
                agent.solve_task(task_id)
                evaluation = agent.world.evaluate().to_dict()
                success = evaluation["success"]
                score = len(evaluation['passes']) / evaluation['num_tests']
                
                results.append({
                    "task_id": task_id,
                    "success": success,
                    "score": score
                })
                
                print(f"      Result: {'✅' if success else '❌'} ({score:.2%})")
                
            except Exception as e:
                print(f"      Error: {e}")
                results.append({
                    "task_id": task_id,
                    "success": False,
                    "score": 0.0,
                    "error": str(e)
                })
            finally:
                if hasattr(agent, 'world'):
                    agent.world.close()
                del agent
        
        # Summary
        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        avg_score = sum(r["score"] for r in results) / total
        
        print(f"Total tasks: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Average score: {avg_score:.2%}")
        
        return {
            "model": model_path,
            "memory": memory_path,
            "dataset": dataset,
            "total": total,
            "successful": successful,
            "success_rate": successful/total,
            "avg_score": avg_score,
            "results": results
        }
        
    finally:
        stop_vllm(vllm_process)

def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model")
    parser.add_argument("--memory", type=str, default=None,
                       help="Path to memory template (optional)")
    parser.add_argument("--dataset", type=str, default="test_normal",
                       help="Dataset to evaluate on")
    parser.add_argument("--max-tasks", type=int, default=None,
                       help="Max number of tasks to evaluate")
    args = parser.parse_args()
    
    # Verify paths
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.memory and not Path(args.memory).exists():
        raise FileNotFoundError(f"Memory template not found: {args.memory}")
    
    results = evaluate(args.model, args.memory, args.dataset, args.max_tasks)
    
    # Save results
    import json
    output_file = f"results_{Path(args.model).name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    main()