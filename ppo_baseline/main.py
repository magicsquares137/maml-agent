

class PPO_LOOP:
	def __init__(self, K: int = 6, random_sample_number: int = 40, difficulties: List[int] = [1,2]):
		# number of rollouts per task
		self.K = K

		# Number of random tasks to pull per iteration
		self.random_sample_number = random_sample_number

		# Difficulty levels to include during training 
		self.difficulties = difficulties

	def collect_rollouts(self):
		pass 

	def get_rewards(self):
		pass 

	def calculate_advantage(self):
		pass 

