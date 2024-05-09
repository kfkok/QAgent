import gymnasium as gym
import numpy as np
from test_env import TestEnv

env = gym.make("CartPole-v1", render_mode="human")

# def to_discreet(state):
#     return np.round(state, 2)

q_values = {}

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation, to_discreet(observation), reward)
    
    if terminated or truncated:
        observation, info = env.reset()
env.close()


