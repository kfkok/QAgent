import numpy as np
import random

class TestEnv:
    def __init__(self, gamma=0.9):
        """
        A simple test environment class representing a finite Markov Decision Process (MDP)
        for reinforcement learning. This environment is used for validating a Q-learning agent.

        The environment consists of states and actions. Taking an action in a state leads to 
        a new state with a certain probability and yields a reward. 

        Attributes:
            action_space (TestActionSpace): An object defining the available actions (0 to n-1).
            current_state (tuple): The current state of the environment.
            done (bool): Whether the episode is finished.
            state_action_table (dict): A dictionary storing state-action value estimates (optional).
            state_table (dict): A dictionary containing information for each state, including:
                * 'trans' (dict, optional): A dictionary mapping actions to possible next states 
                and their probabilities. If None, the episode is done in that state.
                * 'reward' (float): The immediate reward received upon entering the state.
        """
        self.action_space = TestActionSpace(n=2) # 2 possible actions
        self.current_state = (0, 0, 0, 0, 0)
        self.done = False
        self.state_action_table = {}
        self.gamma = gamma

        # The state table represents a 3-layer tree with the following structure:
        # Layer 0: Root node
        # Layer 1: Childs of the root node
        # Layer 2: Grandchilds of the root node
        # The key represents the state and its value contains the transition model and the reward for that state
        # For example, the state (0, 0, 0, 0, 0) has the following transition model:
        # action 0: {(0, 0, 0, 0, 1): 0.2, (0, 0, 0, 1, 0): 0.8} which means that if action 0 is taken, there is a 20% chance of going to state (0, 0, 0, 0, 1) and 80% chance of going to state (0, 0, 0, 1, 0)
        # action 1: {(0, 0, 0, 1, 1): 0.2, (0, 0, 1, 0, 0): 0.8} which means that if action 1 is taken, there is a 20% chance of going to state (0, 0, 0, 1, 1) and 80% chance of going to state (0, 0, 1, 0, 0)
        # The reward for reaching this state is 0
        self.state_table = {
            # ------------ Layer 0 (Root node), The initial state ------------
            (0, 0, 0, 0, 0): {
                'trans': {
                    0: {(0, 0, 0, 0, 1): 0.2, (0, 0, 0, 1, 0): 0.8}, # action: {(next_states, next_states_prob)}
                    1: {(0, 0, 0, 1, 1): 0.2, (0, 0, 1, 0, 0): 0.8}
                }, 
                'reward': 0
            },

            # ------------ Layer 1 (Root node's childs) ------------
            # Child of (0, 0, 0, 0) - Action 0
            (0, 0, 0, 0, 1): {
                'trans':{
                    0: {(0, 0, 1, 0, 1): 0.3, (0, 0, 1, 1, 0): 0.7},
                    1: {(0, 0, 1, 1, 1): 0.2, (0, 1, 0, 0, 0): 0.8}
                }, 
                'reward': 0
            },
            (0, 0, 0, 1, 0): {
                'trans': {
                    0: {(0, 1, 0, 0, 1): 0.2, (0, 1, 0, 1, 0): 0.8},
                    1: {(0, 1, 0, 1, 1): 0.2, (0, 1, 1, 0, 0): 0.8}
                },
                'reward': 0
            },

            # Child of (0, 0, 0, 0) - Action 1
            (0, 0, 0, 1, 1): {
                'trans':{
                    0: {(0, 1, 1, 0, 1): 0.2, (0, 1, 1, 1, 0): 0.8},
                    1: {(0, 1, 1, 1, 1): 0.2, (1, 0, 0, 0, 0): 0.8}
                }, 
                'reward': 0
            },
            (0, 0, 1, 0, 0): {
                'trans': {
                    0: {(1, 0, 0, 0, 1): 0.2, (1, 0, 0, 1, 0): 0.8},
                    1: {(1, 0, 0, 1, 1): 0.2, (1, 0, 1, 0, 0): 0.8}
                }, 
                'reward': 0
            },

            # ------------ Layer 2 (Root node's grandchilds), these nodes are leaf nodes where the episode will end ------------

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

    def get_state_action_value(self, state_probs):
        # state_probs is something like this {(0, 1, 1, 0, 1): 0.2, (0, 1, 1, 1, 0): 0.8}
        # where the key is the next state and the value is the probability of reaching that state
        total_value = 0
        total_prob = sum(state_probs.values())

        for state, prob in state_probs.items():
            reward = self.state_table[state]['reward']

            # If the state has no transition, the state is terminal
            if self.state_table[state]['trans'] is None:
                total_value += (prob / total_prob) * reward

            # Not a terminal state
            else:
                total_value += (prob / total_prob) * (reward + self.gamma * self.get_state_value(state))

        return total_value
    
    def calculate_q_table_state_action_values(self):
        """
        Calculate the state-action values for all states and actions in the environment.
        For each state, the state-action value is calculated as the expected return when taking an action in that state.
        For example, (0, 0, 0, 1, 1): {0: 0.66, 1: 0.72} means that the expected return for taking action 0 in state (0, 0, 0, 1, 1) is 0.66 and for action 1 is 0.72.
        """
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

