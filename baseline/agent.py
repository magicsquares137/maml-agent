# agent.py
from templates import Template
from config import Config
from models import AgentState, Message
from openai import OpenAI
import os
from typing import Dict
from utils import message_parser, truncate_message_history

class ReactAgent:
    def __init__(self, config: Config) -> None:
        self.max_iters: int = config.max_iters  # Fixed: use instance
        self.max_tokens: int = config.max_tokens
        self.temperature: float = config.temperature
        self.base_model: str = config.base_model
        self.truncation_threshold: int = config.truncation_threshold
        self.template = Template()
        self.state = AgentState(max_iters=config.max_iters)
        if config.service == "OpenAI":
            self.client = OpenAI(api_key=config.openai_api_key)
        elif config.service == "TogetherAI":
            os.environ["TOGETHER_API_KEY"] = config.togetherai_api_key
            self.client = OpenAI(
                api_key=config.togetherai_api_key,
                base_url="https://api.together.xyz/v1"
            )
        self.eval_tracker: Dict = {}  
        
    def initialize(
        self, 
        first_name: str, 
        last_name: str, 
        email: str, 
        phone_number: str, 
        task_instructions: str
    ) -> None:
        init_template = self.template.format_prompt(
            first_name, 
            last_name, 
            email, 
            phone_number, 
            task_instructions
        )
        self.state.conversation_history.append(
            Message(role="user", content=init_template)
        )
    
    def call_llm(self):  # Fixed: added colon
        # Truncate if needed
        messages = truncate_message_history(
            self.state.conversation_history, 
            self.truncation_threshold
        )
        
        response = self.client.chat.completions.create(
            model=self.base_model,
            messages=[msg.dict() for msg in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
    
    def step(self, world):  # Fixed: added colon
        """
        Takes conversation, gets LLM response, parses code, executes, updates state
        """
        llm_output = self.call_llm()
        
        # Append LLM response to history
        self.state.conversation_history.append(
            Message(role="assistant", content=llm_output)
        )
        
        # Extract and execute code
        code = message_parser(llm_output)
        observation_string = "No code block found in response."
        
        if code:
            try:
                observation = world.execute(code)
                observation_string = str(observation)
            except Exception as e:
                observation_string = f"Error: {str(e)}"
        
        # Append observation to history
        self.state.conversation_history.append(
            Message(role="user", content=observation_string)
        )
        
        # Update iteration counter
        self.state.iteration += 1
        
        # Check if task completed
        if world.task_completed():
            self.state.done = True
        
        return world
    
    def run(self, world):
        """
        Execute agent loop until completion or max iterations
        """
        while self.state.should_continue:
            world = self.step(world)
            
            if self.state.done:
                print(f"Task completed in {self.state.iteration} iterations")
                break
        
        if not self.state.done:
            print(f"Max iterations ({self.max_iters}) reached without completion")
        
        return world