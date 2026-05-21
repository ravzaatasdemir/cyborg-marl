import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
total_rewards = []

for episode in range(10):
    observation, info = env.reset()
    episode_reward = 0

    for step in range(500):
        # çubuk açısına bak, ona göre karar ver
        angle = observation[2]
        
        if angle > 0:
            action = 1  # sağa eğiliyorsa sağa it
        else:
            action = 0  # sola eğiliyorsa sola it

        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: {episode_reward} adım")

print(f"\nOrtalama: {np.mean(total_rewards):.1f} adım")
env.close()
