import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agents.q_agent import QAgent 

# Example values for the parameters
params = {
    # ------------------------------- #
    # Environment parameters
    # ------------------------------- #

    # The environment to use
    "env_name": "CartPole-v1",

    # ------------------------------- #
    # Agent parameters
    # ------------------------------- #

    # The state filter to use, here we only concern ourselves with the cart velocity, pole angle and pole velocity, thus we ignore the cart position which is the first element in the state
    "state_filter": lambda state : (state[1], state[2], state[3]),

    # The learning rate to use
    "learning_rate": 0.07,

    # The discount factor to use
    "discount_factor": 0.99,

    # The state rounding to use, 1 means we round, for example, 0.3123 to 0.3, this is useful for reducing the state space
    "state_rounding": 1,

    # ------------------------------- #
    # Agent training parameters
    # ------------------------------- #

    # The number of episodes the agent plays during training, the higher the better, but also the longer it takes
    "training_episodes": 7000,

    # The initial epsilon value to use, this is the probability of taking a random action, 1 means always take random actions while 0 means always take the best action
    "initial_epsilon": 1.0,

    # The minimum epsilon value to use, this is the lowest probability of taking a random action
    "min_epsilon": 0.1,

    # The decay percentage to use, this is the percentage by which epsilon decays after each episode from initial_epsilon to min_epsilon
    # For example, if initial_epsilon is 1.0, min_epsilon is 0.1 and decay_percentage is 0.5, then epsilon will decay from 1.0 to 0.1 after 50% of the training episodes
    "decay_percentage": 0.5,

    # ------------------------------- #
    # Test parameters
    # ------------------------------- #

    # The number of episodes the agent plays to test its performance after training
    # During testing, epsilon is set to 0, meaning the agent will always take the best action
    "test_episodes": 20
}

# The run function is called by the main script
def run(params):
    env = gym.make(params["env_name"])
    agent = QAgent(
        env=env,
        state_filter=params["state_filter"],
        learning_rate=params["learning_rate"],
        discount_factor=params["discount_factor"],
        state_rounding=params["state_rounding"]
    )
    reward_over_episodes = []

    # Train the agent for 100 episodes
    agent.train(
        episodes=params["training_episodes"], 
        initial_epsilon=params["initial_epsilon"], 
        min_epsilon=params["min_epsilon"], 
        decay_percentage=params["decay_percentage"]
    )

    # Env mode set to human to visualize the agent's performance
    env = gym.make(params["env_name"], render_mode="human")
    print("Testing the agent...")
    episodes_count = params["test_episodes"]
    for i in range(episodes_count):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_best_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state

        if done:
            reward_over_episodes.append(episode_reward)
            print("Episode terminated, total reward:", episode_reward)

    env.close()

    # Plot the reward over episodes
    plt.plot(reward_over_episodes)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()

run({
    # ------------------------------- #
    # Environment parameters
    # ------------------------------- #

    # The environment to use
    "env_name": "CartPole-v1",

    # ------------------------------- #
    # Agent parameters
    # ------------------------------- #

    # The state filter to use, here we only concern ourselves with the cart velocity, pole angle and pole velocity, thus we ignore the cart position which is the first element in the state
    "state_filter": lambda state : (state[1], state[2], state[3]),

    # The learning rate to use
    "learning_rate": 0.07,

    # The discount factor to use
    "discount_factor": 0.99,

    # The state rounding to use, 1 means we round, for example, 0.3123 to 0.3, this is useful for reducing the state space
    "state_rounding": 1,

    # ------------------------------- #
    # Agent training parameters
    # ------------------------------- #

    # The number of episodes the agent plays during training, the higher the better, but also the longer it takes
    "training_episodes": 11000,

    # The initial epsilon value to use, this is the probability of taking a random action, 1 means always take random actions while 0 means always take the best action
    "initial_epsilon": 1.0,

    # The minimum epsilon value to use, this is the lowest probability of taking a random action
    "min_epsilon": 0.1,

    # The decay percentage to use, this is the percentage by which epsilon decays after each episode from initial_epsilon to min_epsilon
    # For example, if initial_epsilon is 1.0, min_epsilon is 0.1 and decay_percentage is 0.5, then epsilon will decay from 1.0 to 0.1 after 50% of the training episodes
    "decay_percentage": 0.5,

    # ------------------------------- #
    # Test parameters
    # ------------------------------- #

    # The number of episodes the agent plays to test its performance after training
    # During testing, epsilon is set to 0, meaning the agent will always take the best action
    "test_episodes": 20
})