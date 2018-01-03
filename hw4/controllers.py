import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		self.env = env

	def get_action(self, state):
		""" Your code should randomly sample an action uniformly from the action space """
		# TODO(universome): Well, actually we should account for state
		# but looks like it's fine as it is :|
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self,
				 env,
				 dyn_model,
				 horizon=5,
				 cost_fn=None,
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, initial_state):
		"""Note: be careful to batch your simulations through the model for speed"""
		trajectory_states = []
		trajectory_actions = []
		trajectory_next_states = []

		states = [initial_state for _ in range(self.num_simulated_paths)]

		for _ in range(self.horizon):
			actions = [self.env.action_space.sample() for _ in range(self.num_simulated_paths)]
			next_states = self.dyn_model.predict(states, actions)

			trajectory_states.append(states)
			trajectory_actions.append(actions)
			trajectory_next_states.append(next_states)

			states = next_states

		trajectory_states = np.swapaxes(trajectory_states, 0, 1)
		trajectory_actions = np.swapaxes(trajectory_actions, 0, 1)
		trajectory_next_states = np.swapaxes(trajectory_next_states, 0, 1)

		costs = trajectory_cost_fn(trajectory_states, trajectory_actions, trajectory_next_states)

		# print('Min/max cost:', np.min(costs), np.max(costs))

		return trajectory_actions[np.argmin(costs)][0]
