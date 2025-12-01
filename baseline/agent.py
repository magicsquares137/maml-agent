from templates import Template
from config import Config
from models import AgentState, Message
from openai import OpenAI
from typing import Dict
from utils import message_parser, truncate_message_history

class ReactAgent:
	def __init__(
		self, 
		config: Config
		) -> None:
		self.max_iters: int = Config.max_iters
		self.max_tokens: int = Config.max_tokens
		self.temperature: float = Config.temperature
		self.base_model: str = Config.base_model
		self.truncation_threshold: int = Config.truncation_threshold
		self.template = Template()
		self.state = AgentState()
		self.client = OpenAI(api_key=Config.openai_api_key)
		self.eval_tracker = Dict()

	def initialize(
		self, 
		first_name: str, 
		last_name: str, 
		email: str, 
		phone_number: str, 
		task_instructions: str
		) -> None:
		# format the template with input
		init_template = self.template.format_prompt(
			first_name, 
			last_name, 
			email, 
			phone_number, 
			task_instructions
		)

		# make init message and append to conversation history
		self.state.conversation_history.append(
			Message(
				role="user", 
				content=init_template
			)
		)

	def call_llm(self)
		response = self.client.chat.completions.create(
		            model=self.base_model,
		            messages=[
		            	msg.dict() for msg in self.state.conversation_history
		            ],
		            temperature=self.temperature,
		            max_tokens=self.max_tokens
		        )
		return response.choices[0].message.content

	def step(self, world)
		"""
		takes in conversation, gets llm response, appends to state, 
		parses out code, runs code if in there, updates intervals, checks if done
		"""

		# Get current conversation history and llm response
		llm_output = self.call_llm()

		# append input to state
		self.state.conversation_history.append(Message(role="assistant", content=llm_output))

		# Look for code in response
		code = message_parser(llm_output)

		if code:
			try:
				observation = world.execute(code)
				observation_string = str(observation)
				status = "success"
			except Exception as e:
				observation_str = f"Error: {str(e)}"
				status = "error"

		# append observation to conversation history
		self.state.conversation_history.append(Message(role="user", content=observation_string))

		# If task is completed transition to done
		if world.task_completed():
			self.state.done = True

		return world

	def run(self)
		"""
		for each task in provided task set, while not done, iterates through a task

		gets evaluation score after completed
		"""