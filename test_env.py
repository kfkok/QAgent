import numpy as np
import random

class TestEnv:
    def __init__(self):
        self.action_space = TestActionSpace(n=2) # 2 possible actions
        self.current_state = (0, 0, 0, 0, 0)
        self.done = False
        self.state_action_table = {}
        self.state_table = {
            # ------------ Layer 0 ------------
            (0, 0, 0, 0, 0): {
                    'trans': {
                        0: {(0, 0, 0, 0, 1): 0.2, (0, 0, 0, 1, 0): 0.8}, # action: {(next_states, next_states_prob)}
                        1: {(0, 0, 0, 1, 1): 0.2, (0, 0, 1, 0, 0): 0.8}
                    }, 
                    'reward': 0
                },

            # ------------ Layer 1 ------------

            # Child of (0, 0, 0, 0) - Action 0
            (0, 0, 0, 0, 1): {
                'trans':{
                    0: {(0, 0, 1, 0, 1): 0.3, (0, 0, 1, 1, 0): 0.7},
                    1: {(0, 0, 1, 1, 1): 0.2, (0, 1, 0, 0, 0): 0.8}
                }, 
                'reward': 0},
            (0, 0, 0, 1, 0): {
                'trans': {
                    0: {(0, 1, 0, 0, 1): 0.2, (0, 1, 0, 1, 0): 0.8},
                    1: {(0, 1, 0, 1, 1): 0.2, (0, 1, 1, 0, 0): 0.8}
                },
                'reward': 0},

            # Child of (0, 0, 0, 0) - Action 1
            (0, 0, 0, 1, 1): {
                'trans':{
                    0: {(0, 1, 1, 0, 1): 0.2, (0, 1, 1, 1, 0): 0.8},
                    1: {(0, 1, 1, 1, 1): 0.2, (1, 0, 0, 0, 0): 0.8}
                }, 
                'reward': 0},
            (0, 0, 1, 0, 0): {
                'trans': {
                    0: {(1, 0, 0, 0, 1): 0.2, (1, 0, 0, 1, 0): 0.8},
                    1: {(1, 0, 0, 1, 1): 0.2, (1, 0, 1, 0, 0): 0.8}
                }, 
                'reward': 0},

            # ------------ Layer 2 ------------

            # Child of (0, 0, 0, 0, 1) - Action 0
            (0, 0, 1, 0, 1): {'trans': None, 'reward': 1},
            (0, 0, 1, 1, 0): {'trans': None, 'reward': 2},

            # Child of (0, 0, 0, 0, 1) - Action 1
            (0, 0, 1, 1, 1): {'trans': None, 'reward': 0},
            (0, 1, 0, 0, 0): {'trans': None, 'reward': 0},

            # Child of (0, 0, 0, 1, 0) - Action 0
            (0, 1, 0, 0, 1): {'trans': None, 'reward': 10},
            (0, 1, 0, 1, 0): {'trans': None, 'reward': 0},

            # Child of (0, 0, 0, 1, 0) - Action 1
            (0, 1, 0, 1, 1): {'trans': None, 'reward': 0},
            (0, 1, 1, 0, 0): {'trans': None, 'reward': 0},

            # Child of (0, 0, 0, 1, 1) - Action 0
            (0, 1, 1, 0, 1): {'trans': None, 'reward': 20},
            (0, 1, 1, 1, 0): {'trans': None, 'reward': 0},

            # Child of (0, 0, 0, 1, 1) - Action 1
            (0, 1, 1, 1, 1): {'trans': None, 'reward': 9},
            (1, 0, 0, 0, 0): {'trans': None, 'reward': 0},

            # Child of (0, 0, 1, 0, 0) - Action 0
            (1, 0, 0, 0, 1): {'trans': None, 'reward': 0},
            (1, 0, 0, 1, 0): {'trans': None, 'reward': 0},

            # Child of (0, 0, 1, 0, 0) - Action 1
            (1, 0, 0, 1, 1): {'trans': None, 'reward': 0},
            (1, 0, 1, 0, 0): {'trans': None, 'reward': 1},
        }

    def reset(self):
        self.current_state = (0, 0, 0, 0, 0)
        self.done = False  

        return self.current_state, None

    def get_next_state(self, transition):
        states = list(transition.keys()) # The possible next states
        probs = transition.values() # The probability of reaching each next state
        
        return random.choices(states, weights=probs)[0]

    def step(self, action):
        if self.done:
            raise Exception("The episode is done. Call reset() to start a new episode.")

        # The transition model for the current state
        trans = self.state_table[self.current_state]['trans']

        next_state = self.get_next_state(trans[action])
        reward = self.state_table[next_state]['reward']

        # If the next state has no transition, the episode is done
        if self.state_table[next_state]['trans'] is None:
            self.done = True

        self.current_state = next_state
        
        return next_state, reward, self.done, None, None
    
    def get_state_value(self, state):
        trans = self.state_table[state]['trans']
        reward = self.state_table[state]['reward']

        if trans is None:
            return reward
        else:
            return max([self.get_state_action_value(state_probs) for state_probs in trans.values()])

    def get_state_action_value(self, state_probs, gamma=0.9):
        # state_probs is something like this {(0, 1, 1, 0, 1): 0.2, (0, 1, 1, 1, 0): 0.8}
        total_value = 0
        total_prob = sum(state_probs.values())

        for state, prob in state_probs.items():
            reward = self.state_table[state]['reward']
            if self.state_table[state]['trans'] is None:
                total_value += prob * reward
            else:
                total_value += (prob / total_prob) * (reward + gamma * self.get_state_value(state))

        return total_value
    
    def calculate_q_table_state_action_values(self):
        for state, state_info in self.state_table.items():
            trans = state_info['trans']
            if trans is not None:
                self.state_action_table[state] = {action: self.get_state_action_value(state_probs) for action, state_probs in trans.items()}

        return self.state_action_table


class TestActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n-1)

