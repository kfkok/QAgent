import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class QAgent:
    def __init__(self, env, learning_rate=0.01, discount_factor=0.9, epsilon=0.8, state_rounding=2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_rounding = state_rounding
        self.q_table = dict()
        self.env = env
        self.action_count = self.env.action_space.n

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Explore other action
            return self.env.action_space.sample()
        else:
            # Choose best action
            return self.get_best_action(state)
    
    def to_discreet(self, state):
        return tuple(np.round(value, decimals=self.state_rounding) for value in state)

    def get_best_action(self, state):
        """Given a state, we find the action that has the highest value"""
        
        state_actions_values = self.get_state_action_values(state)
        
        return np.argmax(state_actions_values)

    def get_state_action_values(self, state):
        """Given state, we retrieve the all the values for each action"""

        state = self.to_discreet(state)

        # We can't find this state in the table, so let's insert it and randomize its action values
        if not state in self.q_table:
            self.q_table[state] = np.zeros(self.action_count)
        
        return self.q_table[state]
    
    def get_state_value(self, state):
        """Given a state, we get the maximum value that an action can offer"""
        
        state = self.to_discreet(state)

        state_actions_values = self.get_state_action_values(state)

        return np.max(state_actions_values)

    def update_state_action_value(self, state, action, value):
        state = self.to_discreet(state) 

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
        value = q_state_action + self.learning_rate * (q_target - q_state_action) 
        # print("agent: update value =", value)

        self.update_state_action_value(state, action, value)

    def train(self, episodes, initial_epsilon=0.9, min_epsilon=0.1, decay_rate=0.9999, state_filter=None):
        self.epsilon = initial_epsilon
        total_rewards = []  # List to store rewards per episode
        visited_states_count = dict()  # Dictionary to store the number of times each state is visited

        for episode in range(episodes):
            state, info = self.env.reset()  # Reset the environment for each episode
            done = False
            episode_reward = 0
            self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

            while not done:
                # Filter the state if a state_filter function is provided
                filtered_state = state if state_filter is None else state_filter(state)

                # Update the visited_states_count dictionary
                state_key = self.to_discreet(filtered_state)
                visited_states_count[state_key] = visited_states_count.get(state_key, 0) + 1

                action = self.choose_action(filtered_state)
                next_state, reward, done, truncated, info = self.env.step(action)

                # Filter the next state if a state_filter function is provided
                filtered_next_state = next_state if state_filter is None else state_filter(next_state)

                self.update_q_table(filtered_state, action, reward, filtered_next_state, done)
                
                state = next_state

                episode_reward += reward

            total_rewards.append(episode_reward)

            if episode % 10 == 0:  # Print progress update every 10 episodes
                print(f"Episode: {episode}, Total Reward: {episode_reward}", "Epsilon: ", self.epsilon)

        print("Training complete.")

        # Get the values of visited_states_count dictionary
        visit_frequency = {}
        for count in visited_states_count.values():
            visit_frequency[count] = visit_frequency.get(count, 0) + 1

        print("state_visit_frequencies_count = ", visit_frequency)


        # visit_frequency = {}   # New dictionary to track visit frequencies 
        # for count in visited_states_count.values():
        #     visit_frequency[count] = visit_frequency.get(count, 0) + 1

        # x = list(visit_frequency.keys())
        # print("x = ", x)
        # y = list(visit_frequency.values())
        # print("y = ", y)

        # # You can choose either a bar chart or a line plot
        # plt.bar(x, y)  # For a bar chart
        # plt.xlabel("Number of Times Visited")
        # # X axis label should be display in integer
        # plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        # plt.ylabel("Number of States")
        # # Shows the value of each bar
        # for i in range(len(x)):
        #     plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom')
            
        # plt.title("Distribution of State Visit Frequencies")
        # plt.show()

        # Plot rewards after training
        plt.plot(total_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Q-Learning Training Progress")
        plt.show()
   


