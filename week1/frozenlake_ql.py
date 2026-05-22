import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Ortamı oluştur
env = gym.make("FrozenLake-v1", is_slippery=False)

# Q tablosu: 16 durum x 4 aksiyon, hepsi 0'dan başlıyor
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hiperparametreler
learning_rate = 0.8      # ne kadar hızlı öğrensin
discount = 0.95          # gelecekteki ödüllere ne kadar değer versin
epsilon = 1.0            # başta tamamen rastgele
epsilon_decay = 0.995    # her episode'da biraz daha az rastgele
epsilon_min = 0.01
episodes = 1000

rewards_per_episode = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(100):
        # epsilon olasılıkla rastgele, yoksa Q tablosuna göre
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Q tablosunu güncelle
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + learning_rate * (
            reward + discount * next_max - old_value
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

# Sonuçları göster
print(f"İlk 100 episode başarı oranı: {np.mean(rewards_per_episode[:100]):.2%}")
print(f"Son 100 episode başarı oranı: {np.mean(rewards_per_episode[-100:]):.2%}")

# Grafik
plt.plot(np.convolve(rewards_per_episode, np.ones(50)/50, mode='valid'))
plt.title("Q-Learning: FrozenLake Öğrenme Eğrisi")
plt.xlabel("Episode")
plt.ylabel("Ortalama Ödül (50 episode)")
plt.savefig("frozenlake_learning.png")
print("Grafik kaydedildi: frozenlake_learning.png")

env.close()