# # main.py
# import json
# import os
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, List
# from tqdm import tqdm

# from appworld import AppWorld, load_task_ids
# from config import Config
# from agent import ReactAgent

# def ensure_output_dir(experiment_name: str) -> Path:
#     """Create output directory for experiment results"""
#     output_dir = Path(f"experiments/outputs/{experiment_name}")
#     output_dir.mkdir(parents=True, exist_ok=True)
#     return output_dir

# def save_results(
#     eval_tracker: Dict, 
#     output_dir: Path, 
#     dataset_name: str
# ) -> None:
#     """Save evaluation results to JSON using low-level file operations"""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{dataset_name}_results_{timestamp}.json"
#     filepath = str(output_dir / filename)
    
#     # Convert to JSON string
#     json_str = json.dumps(eval_tracker, indent=2)
    
#     # Use low-level os operations to bypass AppWorld's open() wrapper
#     try:
#         # Open file with os.open (not affected by AppWorld's patches)
#         fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
#         os.write(fd, json_str.encode('utf-8'))
#         os.close(fd)
#         print(f"\n💾 Results saved to: {filepath}")
#     except Exception as e:
#         print(f"\n⚠️  Could not save results: {e}")
#         # As fallback, print to console
#         print(f"\n📊 Results (could not save to file):")
#         print(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)

# def calculate_metrics(eval_tracker: Dict) -> Dict:
#     """Calculate TGC and SGC metrics from eval tracker"""
#     total_tasks = len(eval_tracker)
    
#     if total_tasks == 0:
#         return {
#             "total_tasks": 0,
#             "tgc": 0.0,
#             "sgc": 0.0,
#             "completed": 0,
#             "errors": 0
#         }
    
#     # Count successes
#     task_goal_correct = sum(
#         1 for task in eval_tracker.values() 
#         if task.get("result", {}).get("correct", False)
#     )
    
#     subgoal_correct = sum(
#         task.get("result", {}).get("subgoal_correct", 0)
#         for task in eval_tracker.values()
#     )
    
#     total_subgoals = sum(
#         task.get("result", {}).get("total_subgoals", 0)
#         for task in eval_tracker.values()
#     )
    
#     completed = sum(
#         1 for task in eval_tracker.values()
#         if task.get("completed", False)
#     )
    
#     errors = sum(
#         1 for task in eval_tracker.values()
#         if task.get("error") is not None
#     )
    
#     # Calculate percentages
#     tgc = (task_goal_correct / total_tasks) * 100
#     sgc = (subgoal_correct / total_subgoals * 100) if total_subgoals > 0 else 0.0
    
#     return {
#         "total_tasks": total_tasks,
#         "tgc": round(tgc, 1),
#         "sgc": round(sgc, 1),
#         "task_goal_correct": task_goal_correct,
#         "subgoal_correct": subgoal_correct,
#         "total_subgoals": total_subgoals,
#         "completed": completed,
#         "errors": errors,
#         "completion_rate": round((completed / total_tasks) * 100, 1)
#     }

# def print_metrics(metrics: Dict) -> None:
#     """Pretty print evaluation metrics"""
#     print("\n" + "="*60)
#     print("📊 EVALUATION RESULTS")
#     print("="*60)
#     print(f"Total Tasks:          {metrics['total_tasks']}")
#     print(f"Task Goal Correct:    {metrics['task_goal_correct']} ({metrics['tgc']}%)")
#     print(f"Subgoal Correct:      {metrics['subgoal_correct']}/{metrics['total_subgoals']} ({metrics['sgc']}%)")
#     print(f"Completed:            {metrics['completed']} ({metrics['completion_rate']}%)")
#     print(f"Errors:               {metrics['errors']}")
#     print("="*60)
#     print(f"\n🎯 TGC: {metrics['tgc']}%")
#     print(f"🎯 SGC: {metrics['sgc']}%")
#     print("="*60)

# def run_evaluation(
#     dataset_name: str = "test_normal",
#     experiment_name: str = "gpt4o_baseline",
#     max_tasks: int = None,
#     resume_from: str = None
# ) -> Dict:
#     """
#     Run evaluation on specified dataset
    
#     Args:
#         dataset_name: One of ["train", "dev", "test_normal", "test_challenge"]
#         experiment_name: Name for this experiment run
#         max_tasks: Maximum number of tasks to evaluate (None for all)
#         resume_from: Path to previous results JSON to resume from
#     """
    
#     # Setup
#     config = Config()
#     output_dir = ensure_output_dir(experiment_name)
    
#     # Load task IDs
#     task_ids = load_task_ids(dataset_name)
#     if max_tasks:
#         task_ids = task_ids[:max_tasks]
    
#     print(f"\n🚀 Starting evaluation on {dataset_name}")
#     print(f"📝 Total tasks: {len(task_ids)}")
#     print(f"🤖 Model: {config.base_model}")
#     print(f"🔄 Max iterations per task: {config.max_iters}")
    
#     # Resume from previous run if specified
#     eval_tracker = {}
#     completed_task_ids = set()
    
#     if resume_from and os.path.exists(resume_from):
#         with open(resume_from, 'r') as f:
#             eval_tracker = json.load(f)
#         completed_task_ids = set(eval_tracker.keys())
#         print(f"📂 Resuming from: {resume_from}")
#         print(f"✅ Already completed: {len(completed_task_ids)} tasks")
#         task_ids = [tid for tid in task_ids if tid not in completed_task_ids]
    
#     # Run evaluation
#     for idx, task_id in enumerate(tqdm(task_ids, desc="Evaluating tasks")):
#         print(f"\n{'='*60}")
#         print(f"Task {idx + 1}/{len(task_ids)}: {task_id}")
#         print(f"{'='*60}")
        
#         task_result = {
#             "task_id": task_id,
#             "completed": False,
#             "iterations": 0,
#             "error": None,
#             "result": None,
#             "conversation_length": 0
#         }

#         world = None
        
#         try:
#             # Initialize environment and agent
#             with AppWorld(
#                 task_id=task_id,
#                 experiment_name=experiment_name
#             ) as world:
#                 print(f"📋 Instruction: {world.task.instruction}\n")
                
#                 # Create NEW agent for each task
#                 agent = ReactAgent(config)
                
#                 # Initialize with task details
#                 agent.initialize(
#                     first_name=world.task.supervisor.get("first_name", ""),
#                     last_name=world.task.supervisor.get("last_name", ""),
#                     email=world.task.supervisor.get("email", ""),
#                     phone_number=world.task.supervisor.get("phone_number", ""),
#                     task_instructions=world.task.instruction
#                 )
                
#                 # Run agent
#                 world = agent.run(world)
                
#                 # Get results BEFORE leaving the with block
#                 task_result["completed"] = world.task_completed()
#                 task_result["iterations"] = agent.state.iteration
#                 task_result["conversation_length"] = len(agent.state.conversation_history)
                
#                 # Get evaluation result from AppWorld
#                 if hasattr(world, 'get_result'):
#                     task_result["result"] = world.get_result()
#                 elif hasattr(world.task, 'result'):
#                     task_result["result"] = world.task.result
                
#                 print(f"\n✅ Task completed: {task_result['completed']}")
#                 print(f"🔄 Iterations: {task_result['iterations']}")
            
#             # The 'with' block automatically closes world here
                
#         except KeyboardInterrupt:
#             print("\n⚠️  Interrupted by user")
#             # Save what we have so far
#             save_results(eval_tracker, output_dir, f"{dataset_name}_interrupted")
#             raise  # Re-raise to stop execution
            
#         except Exception as e:
#             task_result["error"] = str(e)
#             print(f"\n❌ Error: {e}")
#             import traceback
#             traceback.print_exc()
        
#         # Store result (after with block has closed)
#         eval_tracker[task_id] = task_result
        
#         # Save intermediate results every 10 tasks
#         if (idx + 1) % 10 == 0:
#             save_results(eval_tracker, output_dir, f"{dataset_name}_checkpoint")
#             metrics = calculate_metrics(eval_tracker)
#             print(f"\n📊 Intermediate metrics (after {idx + 1} tasks):")
#             print(f"   TGC: {metrics['tgc']}% | SGC: {metrics['sgc']}%")
    
#     # Save final results
#     save_results(eval_tracker, output_dir, dataset_name)
    
#     # Calculate and print final metrics
#     metrics = calculate_metrics(eval_tracker)
#     print_metrics(metrics)
    
#     return eval_tracker

# def main():
#     """Main entry point"""
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Run AppWorld evaluation")
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default="test_normal",
#         choices=["train", "dev", "test_normal", "test_challenge"],
#         help="Dataset split to evaluate on"
#     )
#     parser.add_argument(
#         "--experiment",
#         type=str,
#         default="gpt4o_baseline",
#         help="Experiment name"
#     )
#     parser.add_argument(
#         "--max-tasks",
#         type=int,
#         default=None,
#         help="Maximum number of tasks to evaluate (for testing)"
#     )
#     parser.add_argument(
#         "--resume-from",
#         type=str,
#         default=None,
#         help="Path to previous results JSON to resume from"
#     )
    
#     args = parser.parse_args()
    
#     # Run evaluation
#     eval_tracker = run_evaluation(
#         dataset_name=args.dataset,
#         experiment_name=args.experiment,
#         max_tasks=args.max_tasks,
#         resume_from=args.resume_from
#     )
    
#     print("\n✨ Evaluation complete!")

# if __name__ == "__main__":
#     main()

# main.py
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from appworld import AppWorld, load_task_ids
from config import Config
from agent import ReactAgent

def ensure_output_dir(experiment_name: str) -> Path:
    """Create output directory for experiment results"""
    output_dir = Path(f"experiments/outputs/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_results(
    eval_tracker: Dict, 
    output_dir: Path, 
    dataset_name: str
) -> None:
    """Save evaluation results to JSON using low-level file operations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_results_{timestamp}.json"
    filepath = str(output_dir / filename)
    
    # Convert to JSON string
    json_str = json.dumps(eval_tracker, indent=2)
    
    # Use low-level os operations to bypass AppWorld's open() wrapper
    try:
        fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(fd, json_str.encode('utf-8'))
        os.close(fd)
        print(f"\n💾 Results saved to: {filepath}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

def calculate_metrics(eval_tracker: Dict) -> Dict:
    """Calculate TGC and SGC metrics from eval tracker"""
    total_tasks = len(eval_tracker)
    
    if total_tasks == 0:
        return {
            "total_tasks": 0,
            "tgc": 0.0,
            "sgc": 0.0,
            "completed": 0,
            "errors": 0
        }
    
    task_goal_correct = sum(
        1 for task in eval_tracker.values() 
        if task.get("result", {}).get("correct", False)
    )
    
    subgoal_correct = sum(
        task.get("result", {}).get("subgoal_correct", 0)
        for task in eval_tracker.values()
    )
    
    total_subgoals = sum(
        task.get("result", {}).get("total_subgoals", 0)
        for task in eval_tracker.values()
    )
    
    completed = sum(
        1 for task in eval_tracker.values()
        if task.get("completed", False)
    )
    
    errors = sum(
        1 for task in eval_tracker.values()
        if task.get("error") is not None
    )
    
    tgc = (task_goal_correct / total_tasks) * 100
    sgc = (subgoal_correct / total_subgoals * 100) if total_subgoals > 0 else 0.0
    
    return {
        "total_tasks": total_tasks,
        "tgc": round(tgc, 1),
        "sgc": round(sgc, 1),
        "task_goal_correct": task_goal_correct,
        "subgoal_correct": subgoal_correct,
        "total_subgoals": total_subgoals,
        "completed": completed,
        "errors": errors,
        "completion_rate": round((completed / total_tasks) * 100, 1)
    }

def print_metrics(metrics: Dict) -> None:
    """Pretty print evaluation metrics"""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Total Tasks:          {metrics['total_tasks']}")
    print(f"Task Goal Correct:    {metrics['task_goal_correct']} ({metrics['tgc']}%)")
    print(f"Subgoal Correct:      {metrics['subgoal_correct']}/{metrics['total_subgoals']} ({metrics['sgc']}%)")
    print(f"Completed:            {metrics['completed']} ({metrics['completion_rate']}%)")
    print(f"Errors:               {metrics['errors']}")
    print("="*60)
    print(f"\n🎯 TGC: {metrics['tgc']}%")
    print(f"🎯 SGC: {metrics['sgc']}%")
    print("="*60)

def run_evaluation(
    dataset_name: str = "test_normal",
    experiment_name: str = "gpt4o_baseline",
    max_tasks: int = None,
    config: Config = None
) -> Dict:
    """Run evaluation on specified dataset"""
    
    if config is None:
        config = Config()
    
    output_dir = ensure_output_dir(experiment_name)
    
    # Load task IDs
    task_ids = load_task_ids(dataset_name)
    if max_tasks:
        task_ids = task_ids[:max_tasks]
    
    print(f"\n🚀 Starting evaluation on {dataset_name}")
    print(f"📝 Total tasks: {len(task_ids)}")
    print(f"🤖 Model: {config.base_model}")
    print(f"🔄 Max iterations per task: {config.max_iters}")
    
    eval_tracker = {}
    
    # For each task in the dataset split
    for index, task_id in enumerate(tqdm(task_ids, desc="Evaluating tasks")):
        print(f"\n{'='*60}")
        print(f"Task {index + 1}/{len(task_ids)}: {task_id}")
        print(f"{'='*60}")
        
        task_result = {
            "task_id": task_id,
            "completed": False,
            "iterations": 0,
            "error": None,
            "result": None,
            "conversation_length": 0
        }
        
        try:
            # Load the appworld environment for the task
            with AppWorld(
                task_id=task_id,
                experiment_name=experiment_name,
            ) as world:
                print(f"📋 Instruction: {world.task.instruction}\n")
                
                # Create agent with the task to solve
                agent = ReactAgent(config)
                agent.initialize(
                    first_name=world.task.supervisor.get("first_name", ""),
                    last_name=world.task.supervisor.get("last_name", ""),
                    email=world.task.supervisor.get("email", ""),
                    phone_number=world.task.supervisor.get("phone_number", ""),
                    task_instructions=world.task.instruction
                )
                
                # Run agent
                agent.run(world)
                
                # Collect results
                task_result["completed"] = world.task_completed()
                task_result["iterations"] = agent.state.iteration
                task_result["conversation_length"] = len(agent.state.conversation_history)
                
                if hasattr(world, 'get_result'):
                    task_result["result"] = world.get_result()
                elif hasattr(world.task, 'result'):
                    task_result["result"] = world.task.result
                
                print(f"\n✅ Task completed: {task_result['completed']}")
                print(f"🔄 Iterations: {task_result['iterations']}")
                
        except Exception as e:
            task_result["error"] = str(e)
            print(f"\n❌ Error: {e}")
        
        # Store result
        eval_tracker[task_id] = task_result
        
        # Save checkpoint every 10 tasks
        if (index + 1) % 10 == 0:
            save_results(eval_tracker, output_dir, f"{dataset_name}_checkpoint")
            metrics = calculate_metrics(eval_tracker)
            print(f"\n📊 Checkpoint (after {index + 1} tasks): TGC={metrics['tgc']}% SGC={metrics['sgc']}%")
    
    # Save final results
    save_results(eval_tracker, output_dir, dataset_name)
    
    # Print final metrics
    metrics = calculate_metrics(eval_tracker)
    print_metrics(metrics)
    
    return eval_tracker

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AppWorld evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_normal",
        choices=["train", "dev", "test_normal", "test_challenge"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="gpt4o_baseline",
        help="Experiment name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4o",
        choices=["gpt4o", "qwen", "mistral"],
        help="Model to use"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate (for testing)"
    )
    
    args = parser.parse_args()
    
    # Select config based on model
    config = Config()
    
    # Run evaluation
    run_evaluation(
        dataset_name=args.dataset,
        experiment_name=args.experiment,
        max_tasks=args.max_tasks,
        config=config
    )
    
    print("\n✨ Evaluation complete!")

if __name__ == "__main__":
    main()