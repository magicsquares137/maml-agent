# # agent.py
# from templates import Template
# from config import Config
# from models import AgentState, Message
# from openai import OpenAI
# import os
# from typing import Dict
# from utils import message_parser, truncate_message_history

# class ReactAgent:
#     def __init__(self, config: Config, return_log_probs: bool = False, seed: int = None) -> None:
#         self.max_iters: int = config.max_iters  # Fixed: use instance
#         self.max_tokens: int = config.max_tokens
#         self.temperature: float = config.temperature
#         self.base_model: str = config.base_model
#         self.truncation_threshold: int = config.truncation_threshold
#         self.template = Template()
#         self.state = AgentState(max_iters=config.max_iters)
#         self.seed = seed
#         if config.service == "OpenAI":
#             self.client = OpenAI(api_key=config.openai_api_key)
#         elif config.service == "TogetherAI":
#             os.environ["TOGETHER_API_KEY"] = config.togetherai_api_key
#             self.client = OpenAI(
#                 api_key=config.togetherai_api_key,
#                 base_url="https://api.together.xyz/v1"
#             )
#         elif config.service == "vLLM":
#             openai_api_key = "EMPTY"
#             openai_api_base = "https://3f4bdsdpetv6x5-8000.proxy.runpod.net/v1"
#             self.client = OpenAI(
#                 api_key=openai_api_key,
#                 base_url=openai_api_base,
#             )
#         self.eval_tracker: Dict = {}  

        
#     def initialize(
#         self, 
#         first_name: str, 
#         last_name: str, 
#         email: str, 
#         phone_number: str, 
#         task_instructions: str
#     ) -> None:
#         init_template = self.template.format_prompt(
#             first_name, 
#             last_name, 
#             email, 
#             phone_number, 
#             task_instructions
#         )
#         self.state.conversation_history.append(
#             Message(role="user", content=init_template)
#         )
    
#     def call_llm(self, return_log_probs: bool = False) -> Tuple[str, Union[None, List[Tuple]]]: 
#         messages = truncate_message_history(
#             self.state.conversation_history, 
#             self.truncation_threshold
#         )

#         extra_args = {}
#         if self.seed:
#             extra_args["seed"] = self.seed
#         if return_log_probs:
#             extra_args["logprobs"] = True
#             extra_args["top_logprobs"] = 1  # only need the generated token

#         response = self.client.chat.completions.create(
#             model=self.base_model,
#             messages=[msg.dict() for msg in messages],
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#             **extra_args,
#         )

#         choice = response.choices[0]
#         text = choice.message.content

#         if not return_log_probs:
#             return text, None

#         token_logprobs = []
#         if choice.logprobs is not None:
#             for item in choice.logprobs.content:
#                 token_logprobs.append((item.token, item.logprob))

#         return text, token_logprobs
    
#     def step(self, world):  
#         """
#         Takes conversation, gets LLM response, parses code, executes, updates state
#         """
#         llm_output, token_logprobs = self.call_llm(return_log_probs=True)
        
#         # Append LLM response to history
#         self.state.conversation_history.append(
#             Message(role="assistant", content=llm_output, log_probs=token_logprobs)
#         )
        
#         # Extract and execute code
#         code = message_parser(llm_output)
#         observation_string = "No code block found in response."
        
#         if code:
#             try:
#                 observation = world.execute(code)
#                 observation_string = str(observation)
#             except Exception as e:
#                 observation_string = f"Error: {str(e)}"
        
#         # Append observation to history
#         self.state.conversation_history.append(
#             Message(role="user", content=observation_string)
#         )
        
#         # Update iteration counter
#         self.state.iteration += 1
        
#         # Check if task completed
#         if world.task_completed():
#             self.state.done = True
        
#         return world
    
#     def run(self, world):
#         """
#         Execute agent loop until completion or max iterations
#         """
#         while self.state.should_continue:
#             world = self.step(world)
            
#             if self.state.done:
#                 print(f"Task completed in {self.state.iteration} iterations")
#                 break
        
#         if not self.state.done:
#             print(f"Max iterations ({self.max_iters}) reached without completion")
        
#         return world

from typing import Union
class ReactAgent:
    def __init__(self, config: Config, return_log_probs: bool = False, seed: int = None) -> None:
        self.max_iters: int = config.max_iters  # Fixed: use instance
        self.max_tokens: int = 512
        self.temperature: float = config.temperature
        self.base_model: str = config.base_model
        self.truncation_threshold: int = config.truncation_threshold
        self.template = Template()
        self.state = AgentState(max_iters=config.max_iters)
        self.seed = seed
        if config.service == "OpenAI":
            self.client = OpenAI(api_key=config.openai_api_key)
        elif config.service == "TogetherAI":
            os.environ["TOGETHER_API_KEY"] = config.togetherai_api_key
            self.client = OpenAI(
                api_key=config.togetherai_api_key,
                base_url="https://api.together.xyz/v1"
            )
        elif config.service == "vLLM":
            openai_api_key = "EMPTY"
            openai_api_base = "https://3f4bdsdpetv6x5-8000.proxy.runpod.net/v1"
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
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
    
    def call_llm(self, return_log_probs: bool = True) -> Tuple[str, Union[None, List[Tuple]]]: 
        messages = truncate_message_history(
            self.state.conversation_history, 
            self.truncation_threshold
        )

        extra_args = {}
        if self.seed:
            extra_args["seed"] = self.seed
        if return_log_probs:
            extra_args["logprobs"] = True
            extra_args["top_logprobs"] = 1  # only need the generated token

        response = self.client.chat.completions.create(
            model=self.base_model,
            messages=[msg.dict() for msg in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **extra_args,
        )

        choice = response.choices[0]
        text = choice.message.content

        if not return_log_probs:
            return text, None

        token_logprobs = []
        if choice.logprobs is not None:
            for item in choice.logprobs.content:
                token_logprobs.append((item.token, item.logprob)) # Note, unlike pytorch inference, logprobs from vllm are not shifted. 

        return text, token_logprobs
    
    def step(self, world):  
        llm_output, token_logprobs = self.call_llm(return_log_probs=True)
        
        # Log full output for analysis
        self.eval_tracker[f"iter_{self.state.iteration}_full_output"] = llm_output
        
        # Extract code and find its position
        code, code_start, code_end = message_parser_with_position(llm_output)
        observation_string = "No code block found in response."
        
        if code:
            try:
                observation = world.execute(code)
                observation_string = str(observation)
            except Exception as e:
                observation_string = f"Error: {str(e)}"
            
            truncated_logprobs = []
            found_opening = False
            backtick_count = 0
            
            for i, (token, logprob) in enumerate(token_logprobs):
                truncated_logprobs.append((token, logprob))
                
                # Count backtick tokens
                if token == '```':
                    backtick_count += 1
                    if backtick_count == 1:
                        found_opening = True
                    elif backtick_count == 2 and found_opening:
                        # Found closing backticks - stop here
                        break
            
            self.state.conversation_history.append(
                Message(role="assistant", content=llm_output[:code_end].strip(), log_probs=truncated_logprobs)
            )
        else:
            # No code found - store full response
            self.state.conversation_history.append(
                Message(role="assistant", content=llm_output, log_probs=token_logprobs)
            )
        
        # Append real observation
        self.state.conversation_history.append(
            Message(role="user", content=f"Output:\n```\n{observation_string}\n```")
        )
        
        self.state.iteration += 1
        
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
