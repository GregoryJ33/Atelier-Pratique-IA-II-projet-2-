import gymnasium as gym
import ale_py
import numpy as np



env = gym.make("ALE/TicTacToe3Dici-v5", render_mode="human")

def run_random_episode(env):
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    return total_reward

for i in range(5):
    reward = run_random_episode(env)
    print(f"Essai {i} - Total reward: {reward}")