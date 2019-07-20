"""
Iterated Prisoner's dilemma environment.
"""

import gym
import numpy as np

from gym.spaces import Discrete, Tuple

class IteratedPrisonnersDilemma(gym.Env):
    
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5
    
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.payout_matrix = np.array([[-1.,0.],[-3.,-2.]]) # payout matrices as nr of  prison years to serve
        self.action_space = Tuple([Discrete(self.NUM_ACTIONS), Discrete(self.NUM_ACTIONS)])
        self.observation_space = Tuple([self.NUM_STATES, self.NUM_STATES])
        self.step_count = None
        
    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.NUM_STATES)
        init_state[-1] = 1
        observations = [init_state, init_state]
        return observations
    
    def step(self, action):
        action1, action2 = action
        self.step_count += 1
        rewards = [self.payout_matrix[action1, action2], self.payout_matrix[action2, action1]]
        state = np.zeros(self.NUM_STATES)
        state[action1 * 2 + action2] = 1
        observations = [state, state]
    
        done = (self.step_count == self.max_steps)
        
        return observations, rewards, done