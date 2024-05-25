import numpy as np
import matplotlib.pyplot as plt

class QAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.9, state_rounding=1, state_filter=None):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.state_rounding = state_rounding
        self.q_table = dict()
        self.env = env
        self.action_count = self.env.action_space.n
        self.state_filter = state_filter

    def get_action(self, state, state_processed=False, epsilon=0.9):
        """
        Given a state, we choose an action to take based on the epsilon-greedy policy
        The higher the epsilon, the more likely we are to explore other actions
        """
        if np.random.rand() < epsilon:
            # Explore other action
            return self.env.action_space.sample()
        else:
            # Choose best action
            return self.get_best_action(state, state_processed)
    
    def to_discreet(self, state):
        return tuple(np.round(value, decimals=self.state_rounding) for value in state)

    def get_best_action(self, state, state_processed=False):
        """Given a state, we find the action that has the highest value"""
        
        if not state_processed:
            state = self.state_filter(state) if self.state_filter is not None else state
            state = self.to_discreet(state)

        state_actions_values = self.get_state_action_values(state)
        
        return np.argmax(state_actions_values)

    def get_state_action_values(self, state):
        """Given state, we retrieve the all the values for each action"""

        # We can't find this state in the table, so let's insert it and randomize its action values
        if not state in self.q_table:
            self.q_table[state] = np.zeros(self.action_count)
        
        return self.q_table[state]
    
    def get_state_value(self, state):
        """Given a state, we get the maximum value that an action can offer"""
        
        state_actions_values = self.get_state_action_values(state)

        return np.max(state_actions_values)

    def update_state_action_value(self, state, action, value):
        self.q_table[state][action] = value

    def update_q_table(self, state, action, reward, next_state, done):

        # We get the state_action value for the current state and action 
        q_state_action = self.get_state_action_values(state)[action]
        # print("agent: q_state_action =", q_state_action)

        # We get the value for the next state
        q_next_state = self.get_state_value(next_state)
        # print("agent: q_next_state =", q_next_state)
        
        if done:
            q_target = reward
        else:
            q_target = reward + self.discount_factor * (q_next_state)

        # print("agent: q_target =", q_target)

        # Update the state-action value
        new_q_state_action = q_state_action + self.learning_rate * (q_target - q_state_action) 
        # print("agent: update value =", value)

        self.update_state_action_value(state, action, new_q_state_action)

    def plot_state_visit_frequencies(self, state_visit_count):
        # print("state_visit_frequencies_count = ", state_visit_count)
        
        visit_frequency = {}   # New dictionary to track visit frequencies 
        for count in state_visit_count.values():
            visit_frequency[count] = visit_frequency.get(count, 0) + 1

        # Order the dictionary by key
        visit_frequency = dict(sorted(visit_frequency.items()))
        # print("visit_frequency = ", visit_frequency)

        x = list(visit_frequency.keys())
        y = list(visit_frequency.values())

        # Generate positions for bars
        positions = np.arange(len(x))

        plt.bar(positions, y)  # For a bar chart
        plt.xlabel("Number of Times Visited")
        plt.ylabel("Number of States")
        plt.title("Distribution of State Visit Frequencies")
        plt.xticks(positions, x)  # Set the x-axis ticks to match positions, with labels from x
        plt.show()

    def train(self, episodes, initial_epsilon=1.0, min_epsilon=0.5, decay_percentage=0.5):
        """Train the agent using Q-Learning algorithm"""
        """episodes: Number of episodes to train the agent"""
        """initial_epsilon: Initial epsilon value for epsilon-greedy policy"""
        """min_epsilon: Minimum epsilon value for epsilon-greedy policy"""
        """decay_percentage: Normalized (0 to 1) percentage of episodes to decay epsilon value to minimum epsilon"""
        
        epsilon = initial_epsilon
        rewards_over_episodes = []  # List to store rewards per episode
        state_visit_count = dict()  # Dictionary to store the number of times each state is visited

        print("Training started...")
        print("Initial Epsilon: ", initial_epsilon)
        print("Minimum Epsilon: ", min_epsilon)
        print("Decay Percentage: ", decay_percentage)
        print("State Rounding: ", self.state_rounding)
        print("Learning Rate: ", self.learning_rate)
        print("Discount Factor: ", self.discount_factor)
        print("State Visit Count:", state_visit_count)

        decay_rate = (min_epsilon / initial_epsilon) ** (1.0 / (decay_percentage * episodes))

        for episode in range(episodes):
            state, info = self.env.reset()  # Reset the environment for each episode
            done = False
            episode_reward = 0
            discrete_next_state = None

            # Decay the epsilon value
            epsilon = max(min_epsilon, initial_epsilon * decay_rate ** (episode))
            
            while not done:
                if not discrete_next_state:                 
                    # Filter the state if a state_filter function is provided
                    filtered_state = state if self.state_filter is None else self.state_filter(state)
                    discreet_state = self.to_discreet(filtered_state)
                else:
                    discreet_state = discrete_next_state

                # Update the visited_states_count dictionary
                state_visit_count[discreet_state] = state_visit_count.get(discreet_state, 0) + 1

                # Choose an action to be taken under this state
                action = self.get_action(discreet_state, state_processed=True, epsilon=epsilon)
                next_state, reward, done, truncated, info = self.env.step(action)

                # Filter the next state if a state_filter function is provided
                filtered_next_state = next_state if self.state_filter is None else self.state_filter(next_state)
                discrete_next_state = self.to_discreet(filtered_next_state)

                if done:
                    state_visit_count[discrete_next_state] = state_visit_count.get(discrete_next_state, 0) + 1

                self.update_q_table(discreet_state, action, reward, discrete_next_state, done)
                
                episode_reward += reward

            rewards_over_episodes.append(episode_reward)

            if episode % 10 == 0:  # Print progress update every 10 episodes
                print(f"Episode: {episode}, Total Reward: {episode_reward}", "Epsilon: ", epsilon)

        print("Training complete.")

        self.plot_state_visit_frequencies(state_visit_count)

        # Plot rewards after training
        plt.plot(rewards_over_episodes)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Q-Learning Training Progress")
        plt.show()
   


