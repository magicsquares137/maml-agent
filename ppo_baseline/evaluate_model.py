# evaluate_model.py
import os
from dotenv import load_dotenv, find_dotenv
# Load .env + set APPWORLD_ROOT BEFORE importing appworld (it resolves the root
# at import time and defaults to cwd otherwise).
load_dotenv(find_dotenv())
if os.getenv("APPWORLD_ROOT"):
    os.environ["APPWORLD_ROOT"] = os.getenv("APPWORLD_ROOT")

import argparse
import subprocess
import time
import signal
import json
import requests
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from appworld import load_task_ids

# The appworld agent requires these env vars (fill_model_server_url reads
# MODEL_SERVER_URL even when base_url has no template).
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("NO_API_KEY", "EMPTY")
os.environ.setdefault("MODEL_SERVER_URL", "http://localhost:8000")

VLLM_BIN = os.environ.get("VLLM_BIN", "vllm")


def start_vllm(model_path: str, port: int = 8000, tensor_parallel: int = 2,
               gpu_mem_util: float = 0.90, max_num_seqs: int = 8, max_model_len: int = 32768):
    """Start vLLM server. Defaults sized for an 8B on 2x16GB cards (TP=2)."""
    cmd = [
        VLLM_BIN, "serve", model_path,
        "--host", "localhost",
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--tensor-parallel-size", str(tensor_parallel),
        "--max-num-seqs", str(max_num_seqs),
    ]

    log_file = open(f"vllm_eval_{port}.log", "wb")  # truncate per run (avoid stale-log confusion)
    process = subprocess.Popen(
        cmd, stdout=log_file, stderr=log_file, start_new_session=True
    )

    print(f"   Starting vLLM (PID {process.pid}) on port {port}...")
    # 900s: weights live on the slow USB Expansion HDD, so 16GB load + TP=2 shard
    # + compile can take several minutes on a cold read.
    for i in range(900):
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=1)
            if r.status_code == 200:
                print(f"   ✅ vLLM ready on port {port}!")
                return process
        except:
            pass
        time.sleep(1)
        if i % 10 == 0 and i > 0:
            print(".", end="", flush=True)

    raise RuntimeError("vLLM failed to start")


def stop_vllm(process, port: int = 8000):
    """Stop vLLM server (session-scoped kill so we don't hit other arms)."""
    print(f"\n   Stopping vLLM on port {port}...")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        time.sleep(3)
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except:
        pass
    time.sleep(8)


def create_prompt_with_memory(memory_path):
    """Create prompt file with optional memory template."""
    original_prompt = os.environ.get(
        "APPWORLD_PROMPT_FILE",
        os.path.join(os.environ.get("APPWORLD_ROOT", "."),
                     "experiments/prompts/react_code_agent/instructions.txt"),
    )
    if not memory_path:
        return original_prompt
    with open(memory_path, 'r') as f:
        memory = f.read()
    with open(original_prompt, 'r') as f:
        original = f.read()
    combined = f"""### BEST PRACTICES:

{memory}

### END BEST PRACTICES
---

{original}
"""
    temp_path = "./temp_prompt_with_memory.txt"
    with open(temp_path, 'w') as f:
        f.write(combined)
    return temp_path


def _eval_worker(unit: dict) -> dict:
    """Solve ONE task in an isolated AppWorld DB (unique experiment_name) and
    return its score. Module-level so it is picklable for spawn workers. Mirrors
    the training rollout worker's isolation, but with greedy eval settings
    (temp 0.0, no logprobs) and the paper's 50-interaction eval budget."""
    import os
    from appworld import AppWorld
    from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("NO_API_KEY", "EMPTY")
    os.environ.setdefault("MODEL_SERVER_URL", unit["vllm_url"])

    expname = unit["experiment_name"]
    agent = SimplifiedReActCodeAgent(
        model_config={
            "client_name": "openai",
            "api_type": "chat_completions",
            "base_url": unit["vllm_url"],
            "name": unit["model_name"],
            "api_key_env_name": "NO_API_KEY",
            "temperature": 0.0,          # greedy eval (matches baseline Config.temperature=0.0)
            "seed": 100,
            "max_completion_tokens": unit["max_tokens"],
            "cost_per_token": {
                "input_cache_hit": 0.0, "input_cache_miss": 0.0,
                "input_cache_write": 0.0, "output": 0.0,
            },
            "use_cache": False,
            "retry_after_n_seconds": 15,
            "max_retries": 100,
        },
        logger_config={"color": False, "verbose": False},
        appworld_config={"random_seed": 100},
        prompt_file_path=unit["prompt_path"],
        ignore_multiple_calls=True,
        max_prompt_length=None,
        max_output_length=None,
        max_steps=unit["max_steps"],
    )

    res = {"task_id": unit["task_id"], "success": False, "score": 0.0, "error": None}
    try:
        agent.logger.initialize(expname, 1, 1, 0)
        with AppWorld.initializer(update_defaults=True, experiment_name=expname, random_seed=100):
            agent.solve_task(unit["task_id"])
            evaluation = agent.world.evaluate().to_dict()
        res["success"] = bool(evaluation["success"])
        res["score"] = len(evaluation["passes"]) / evaluation["num_tests"]
    except Exception:
        import traceback
        res["error"] = traceback.format_exc()
    finally:
        del agent
    return res


def evaluate(model_path, memory_path, dataset, max_tasks=None, port=8000,
             workers=8, max_steps=50, max_tokens=2048):
    print(f"\n{'='*80}\nEVALUATION\n{'='*80}")
    print(f"Model: {model_path}\nDataset: {dataset}\nWorkers: {workers}  Port: {port}\n{'='*80}\n")

    vllm_process = start_vllm(model_path, port=port)
    try:
        prompt_path = create_prompt_with_memory(memory_path)
        task_ids = load_task_ids(dataset)
        if max_tasks:
            task_ids = task_ids[:max_tasks]
        print(f"\n   Evaluating {len(task_ids)} tasks with {workers} parallel workers...")

        vllm_url = f"http://localhost:{port}/v1"
        mtag = Path(model_path).name
        units = [
            {
                "task_id": tid,
                "vllm_url": vllm_url,
                "model_name": model_path,
                "experiment_name": f"eval_{mtag}_{i}",
                "prompt_path": prompt_path,
                "max_steps": max_steps,
                "max_tokens": max_tokens,
            }
            for i, tid in enumerate(task_ids)
        ]

        results = []
        if workers <= 1:
            for u in tqdm(units, desc="eval"):
                results.append(_eval_worker(u))
        else:
            # Each worker is a separate process making HTTP calls to the ONE vLLM
            # server (which batches concurrent requests). spawn gives clean procs.
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for r in tqdm(pool.imap_unordered(_eval_worker, units),
                              total=len(units), desc="eval"):
                    results.append(r)

        total = len(results)
        successful = sum(1 for r in results if r["success"])
        avg_score = sum(r["score"] for r in results) / total if total else 0.0
        n_err = sum(1 for r in results if r.get("error"))

        print(f"\n{'='*80}\nRESULTS\n{'='*80}")
        print(f"Total tasks: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Average score: {avg_score:.2%}")
        print(f"Errored tasks: {n_err}")

        return {
            "model": model_path, "dataset": dataset,
            "total": total, "successful": successful,
            "success_rate": successful/total if total else 0.0,
            "avg_score": avg_score, "errored": n_err,
            "results": results,
        }
    finally:
        stop_vllm(vllm_process, port=port)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on AppWorld (parallel)")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--memory", type=str, default=None, help="Optional memory template")
    parser.add_argument("--dataset", type=str, default="test_normal", help="Dataset to eval on")
    parser.add_argument("--max-tasks", type=int, default=None, help="Cap number of tasks")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port")
    parser.add_argument("--workers", type=int, default=8, help="Parallel task workers")
    parser.add_argument("--max-steps", type=int, default=50, help="Interaction budget per task")
    parser.add_argument("--out", type=str, default=None, help="Results JSON path")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.memory and not Path(args.memory).exists():
        raise FileNotFoundError(f"Memory template not found: {args.memory}")

    results = evaluate(args.model, args.memory, args.dataset, args.max_tasks,
                       port=args.port, workers=args.workers, max_steps=args.max_steps)

    output_file = args.out or f"results_{Path(args.model).name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
