import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
observation, info = env.reset()

total_rewards = []

for episode in range(10):
    observation, info = env.reset()
    episode_reward = 0

    for step in range(500):
        action = env.action_space.sample()  # rastgele aksiyon
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: {episode_reward} adım")

print(f"\nOrtalama: {np.mean(total_rewards):.1f} adım")
env.close()