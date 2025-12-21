# debug_failed_task.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from typing import Union, Dict, List, Tuple, Optional
from shared.templates import Template
from shared.config import Config
from shared.models import AgentState, Message
from openai import OpenAI
import os
from shared.utils import message_parser_with_position, truncate_message_history
from shared.config import Config
from shared.agent import ReactAgent
from appworld import AppWorld

# Pick a difficulty 1 task that failed
# From your earlier run: e7a10f8_2 had "Error code: 404"
task_id = "e7a10f8_2"  # Or pick any failed diff=1 task

config = Config()
agent = ReactAgent(config, seed=42)

print(f"\n{'='*60}")
print(f"Debugging Task: {task_id}")
print(f"{'='*60}\n")

with AppWorld(task_id=task_id, experiment_name="debug") as world:
    print(f"📋 Instruction: {world.task.instruction}\n")
    
    agent.initialize(
        first_name=world.task.supervisor.get("first_name", ""),
        last_name=world.task.supervisor.get("last_name", ""),
        email=world.task.supervisor.get("email", ""),
        phone_number=world.task.supervisor.get("phone_number", ""),
        task_instructions=world.task.instruction,
        app_descriptions=world.task.app_descriptions
    )
    
    # Run step by step with detailed logging
    for i in range(min(10, agent.max_iters)):
        print(f"\n{'='*60}")
        print(f"ITERATION {i+1}")
        print(f"{'='*60}\n")
        
        try:
            # Generate response
            llm_output, token_logprobs, prompt_token_ids = agent.call_llm(return_log_probs=False)
            
            print(f"🤖 MODEL OUTPUT ({len(llm_output)} chars):")
            print(llm_output)
            print()
            
            # Parse code
            from shared.utils import message_parser_with_position
            code, start, end = message_parser_with_position(llm_output)
            
            if code:
                print(f"✅ EXTRACTED CODE (pos {start}-{end}):")
                print(code)
                print()
                
                # Execute
                observation = world.execute(code)
                print(f"📤 OBSERVATION:")
                print(observation)
                print()
            else:
                print("⚠️  NO CODE FOUND!")
                print()
                break
            
            # Update state
            agent.state.conversation_history.append(
                Message(role="assistant", content=llm_output[:end].strip() if code else llm_output)
            )
            agent.state.conversation_history.append(
                Message(role="user", content=f"Output:\n```\n{observation}\n```")
            )
            agent.state.iteration += 1
            
            if world.task_completed():
                print("✅ TASK COMPLETED!")
                break
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final evaluation
    eval_result = world.evaluate().to_dict()
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Completed: {world.task_completed()}")
    print(f"Correct: {eval_result.get('correct', False)}")
    print(f"Tests passed: {len(eval_result.get('passes', []))}/{eval_result.get('num_tests', 0)}")