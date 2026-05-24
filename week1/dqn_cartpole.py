import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# ---- 1. SİNİR AĞI ----
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),   # 4 giriş (observation), 128 nöron
            nn.ReLU(),
            nn.Linear(128, 128), # 128 nöron
            nn.ReLU(),
            nn.Linear(128, 2)    # 2 çıkış (sol veya sağ)
        )

    def forward(self, x):
        return self.network(x)

# ---- 2. REPLAY BUFFER ----
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

# ---- 3. AJAN ----
class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.buffer = ReplayBuffer()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 64

    def act(self, state):
        if np.random.random() < self.epsilon:
            return random.randint(0, 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)     

# ---- 4. ANA DÖNGÜ ----
env = gym.make("CartPole-v1")
agent = DQNAgent()
rewards_per_episode = []

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0

    for step in range(500):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.push(state, action, reward, next_state, float(done))
        agent.train()

        state = next_state
        total_reward += reward

        if done:
            break

    rewards_per_episode.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg = np.mean(rewards_per_episode[-50:])
        print(f"Episode {episode+1}: Son 50 ortalama = {avg:.1f}")

# Grafik
plt.plot(np.convolve(rewards_per_episode, np.ones(50)/50, mode='valid'))
plt.title("DQN: CartPole Öğrenme Eğrisi")
plt.xlabel("Episode")
plt.ylabel("Ortalama Ödül (50 episode)")
plt.savefig("dqn_learning.png")
print("Grafik kaydedildi.")
env.close()           