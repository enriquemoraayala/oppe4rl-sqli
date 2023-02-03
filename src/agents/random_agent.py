import random
import numpy as np


class RandomAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
    
    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        return random.choice(np.arange(self.action_size))
