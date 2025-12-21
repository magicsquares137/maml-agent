from typing import Union, Dict, List, Tuple, Optional
from shared.templates import Template
from shared.config import Config
from shared.models import AgentState, Message
from openai import OpenAI
import os
from shared.utils import message_parser_with_position, truncate_message_history


class ReactAgent:
    """
    ReAct-style agent that interacts with an AppWorld environment.
    
    Generates reasoning and code blocks, executes them in the environment,
    and observes the results. Stores token-level information for RL training.
    
    Attributes:
        max_iters: Maximum number of interaction iterations
        max_tokens: Maximum tokens per generation
        temperature: Sampling temperature
        base_model: Model identifier string
        state: Current agent state tracking conversation history
        client: OpenAI-compatible API client
        eval_tracker: Dictionary for storing evaluation metrics
    """
    def __init__(
        self, 
        config: Config, 
        return_log_probs: bool = False, 
        seed: Optional[int] = None,
        lora_adapter_path: Optional[str] = None
    ) -> None:
        """
        Initialize ReactAgent with configuration.
        
        Args:
            config: Configuration object with model and service settings
            return_log_probs: Whether to return log probabilities (deprecated, always True)
            seed: Random seed for reproducible generation
        """
        # For PPO, we train LoRAs so can pass loras direct to vLLM
        self.lora_adapter_path = lora_adapter_path

        # Max iterations to attempt any given appworld task
        self.max_iters: int = config.max_iters  

        # Max output tokens
        self.use_log_probs: str = config.use_log_probs
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
            openai_api_base = config.vllm_url
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
        task_instructions: str,
        app_descriptions: str
    ) -> None:
        """
        Initialize agent with task context and user information.
        
        Creates the initial system message with task instructions and
        user profile information.
        
        Args:
            first_name: User's first name
            last_name: User's last name
            email: User's email address
            phone_number: User's phone number
            task_instructions: Description of the task to accomplish
        """
        init_template = self.template.format_prompt(
            first_name, 
            last_name, 
            email, 
            phone_number, 
            task_instructions,
            app_descriptions
        )

        message = Message(
                role="user", 
                content=init_template, 
            )

        self.state.conversation_history.append(
            message
        )
    
    def call_llm(
        self, 
        return_log_probs: bool = True
    ) -> Tuple[str, Optional[List[Tuple[str, float, int]]], Optional[List[int]]]:
        """
        Call the LLM to generate the next response.
        
        Returns token IDs and log probabilities from vLLM to avoid
        retokenization drift during RL training.
        
        Args:
            return_log_probs: Whether to return log probabilities and token IDs
        
        Returns:
            Tuple containing:
                - text: Generated text response
                - token_logprobs: List of (token_str, logprob, token_id) tuples
                - prompt_token_ids: List of input token IDs
        """
        messages = truncate_message_history(
            self.state.conversation_history, 
            self.truncation_threshold
        )

        prompt_token_ids = None

        extra_args = {}
        if self.seed:
            extra_args["seed"] = self.seed
        if return_log_probs:
            extra_args["logprobs"] = True
            extra_args["top_logprobs"] = 1  # only need the generated token
            
            # Tell vLLM to use the LoRA adapter
            if self.lora_adapter_path:
                extra_args["extra_body"] = {
                    "return_token_ids": True,
                    "lora_request": {
                        "lora_name": "current_policy",
                        "lora_path": self.lora_adapter_path
                    }
                }
            else:
                extra_args["extra_body"] = {"return_token_ids": True}

        if return_log_probs:
            response = self.client.chat.completions.create(
                model=self.base_model,
                messages=[m.dict(exclude={"log_probs", "tokenized_input"}) for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **extra_args,
            )

            choice = response.choices[0]
            text = choice.message.content    
        else:
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
            # In this case we are just baselining, dont need tokens or log probs
            return text, None, None

        # Build list of (token_str, logprob, token_id) tuples
        token_logprobs = []
        output_token_ids = getattr(choice, "token_ids", None)
        if choice.logprobs is not None:
            for i, item in enumerate(choice.logprobs.content):
                token_id = output_token_ids[i] if i < len(output_token_ids) else None
                token_logprobs.append((item.token, item.logprob, token_id))
                # Note, unlike pytorch inference, logprobs from vllm are not shifted by 1 index spot. 
            
        # Extract output token IDs
        prompt_token_ids = getattr(response, 'prompt_token_ids', None)
        
        # output text, output tokens/log probs, input tokens
        return text, token_logprobs, prompt_token_ids 
    
    def step(self, world) -> object:  
        """
        Execute one step of the ReAct loop.
        
        Generates a response, extracts and executes code if present,
        observes the result, and updates conversation history.
        
        Truncates stored outputs to only include content up to the first
        code block to avoid training on hallucinated future interactions.
        
        Args:
            world: AppWorld environment instance
        
        Returns:
            Updated world object after code execution
        """
        if self.use_log_probs == "NO":
            llm_output, token_logprobs, prompt_token_ids = self.call_llm(return_log_probs=False)
        else:
            llm_output, token_logprobs, prompt_token_ids = self.call_llm(return_log_probs=True)
        
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

            # Default: no truncation if we don't have logprobs
            truncated_logprobs = token_logprobs

            # Stop after the closing backticks of the first code block  
            # Truncate log probs to match truncated content only if we have them
            if token_logprobs:
                truncated_logprobs = []
                found_opening = False
                backtick_count = 0
                
                for i, (token, logprob, token_id) in enumerate(token_logprobs):
                    truncated_logprobs.append((token, logprob, token_id))
                    
                    # Count backtick tokens
                    if token == '```':
                        backtick_count += 1
                        if backtick_count == 1:
                            found_opening = True
                        elif backtick_count == 2 and found_opening:
                            # Found closing backticks - stop here
                            break

            # Store truncated response (reasoning + code block only)
            self.state.conversation_history.append( 
                Message(role="assistant", content=llm_output[:code_end].strip(), log_probs=truncated_logprobs, tokenized_input=prompt_token_ids)
            )
        else:
            # No code found - store full response
            self.state.conversation_history.append(
                Message(role="assistant", content=llm_output, log_probs=token_logprobs, tokenized_input=prompt_token_ids)
            )
        
        # Append real observation
        self.state.conversation_history.append(
            Message(role="user", content=f"Output:\n```\n{observation_string}\n```")
        )
        
        self.state.iteration += 1
        
        if world.task_completed():
            self.state.done = True
        
        return world
    
    def run(self, world) -> object:
        """
        Execute the full agent loop until task completion or max iterations.
        
        Args:
            world: AppWorld environment instance
        
        Returns:
            Final world state after agent execution
        """
        while self.state.should_continue:
            world = self.step(world)
            
            if self.state.done:
                print(f"Task completed in {self.state.iteration} iterations")
                break
        
        if not self.state.done:
            print(f"Max iterations ({self.max_iters}) reached without completion")
        
        return world