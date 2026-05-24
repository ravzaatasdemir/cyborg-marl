import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np

# Ortamı oluştur
env = gym.make("CartPole-v1")

# PPO ajanını oluştur
model = PPO(
    "MlpPolicy",  # sinir ağı tipi
    env,
    verbose=1,    # eğitim sürecini yazdır
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

# Eğit
print("Eğitim başlıyor...")
model.learn(total_timesteps=100000)

# Test et
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"\nOrtalama ödül: {mean_reward:.1f} +/- {std_reward:.1f}")

env.close()
