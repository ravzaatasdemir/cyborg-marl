import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=True)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = 0.8
discount = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

rewards_per_episode = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    # SARSA farkı: başta aksiyonu seç
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])

    for step in range(100):
        next_state, reward, terminated, truncated, _ = env.step(action)

        # SARSA farkı: sonraki aksiyonu da şimdi seç
        if np.random.random() < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state])

        # SARSA güncelleme — next_action kullanıyor, np.max değil
        q_table[state, action] += learning_rate * (
            reward + discount * q_table[next_state, next_action] - q_table[state, action]
        )

        state = next_state
        action = next_action
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

print(f"İlk 100 episode başarı oranı: {np.mean(rewards_per_episode[:100]):.2%}")
print(f"Son 100 episode başarı oranı: {np.mean(rewards_per_episode[-100:]):.2%}")

plt.plot(np.convolve(rewards_per_episode, np.ones(50)/50, mode='valid'))
plt.title("SARSA: FrozenLake Öğrenme Eğrisi")
plt.xlabel("Episode")
plt.ylabel("Ortalama Ödül (50 episode)")
plt.savefig("sarsa_learning.png")
print("Grafik kaydedildi: sarsa_learning.png")

env.close()