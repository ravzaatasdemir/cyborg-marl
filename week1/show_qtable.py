import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = 0.8
discount = 0.95
epsilon = 1.0

for episode in range(1000):
    state, _ = env.reset()
    for step in range(100):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        q_table[state, action] += learning_rate * (
            reward + discount * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state
        if terminated or truncated:
            break
    epsilon = max(0.01, epsilon * 0.995)

print("Q Tablosu (16 durum x 4 aksiyon):")
print(np.round(q_table, 2))
print()
print("Her durumda en iyi aksiyon (0=Sol 1=Aşağı 2=Sağ 3=Yukarı):")
print(np.argmax(q_table, axis=1).reshape(4, 4))