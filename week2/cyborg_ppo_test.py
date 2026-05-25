from CybORG import CybORG
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers import PettingZooParallelWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SingleAgentWrapper(gym.Env):
    def __init__(self):
        sg = DroneSwarmScenarioGenerator()
        cyborg = CybORG(sg, 'sim')
        self.env = PettingZooParallelWrapper(env=cyborg)
        self.agent = 'blue_agent_0'
        obs_space = self.env.observation_space(self.agent)
        act_space = self.env.action_space(self.agent)
        self.observation_space = gym.spaces.Box(
            low=0, high=200, shape=obs_space.shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(act_space.n)

    def reset(self, seed=None, options=None):
        observations = self.env.reset()
        return observations[self.agent].astype(np.float32), {}

    def step(self, action):
        actions = {}
        for agent in self.env.agents:
            if agent == self.agent:
                actions[agent] = action
            else:
                actions[agent] = self.env.action_space(agent).sample()
        observations, rewards, terminations, infos = self.env.step(actions)
        obs = observations.get(self.agent, np.zeros(self.observation_space.shape, dtype=np.float32))
        reward = rewards.get(self.agent, 0.0)
        done = terminations.get(self.agent, False)
        return obs.astype(np.float32), float(reward), done, False, {}

# Modeli yükle
print("Model yükleniyor...")
env = DummyVecEnv([SingleAgentWrapper])
model = PPO.load("cyborg_ppo_model", env=env)

# Test et
print("Test başlıyor...\n")
episodes = 5
steps = 50

for episode in range(episodes):
    obs = env.reset()
    total_reward = 0

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        if done[0]:
            break

    print(f"Episode {episode+1}: Toplam ödül = {total_reward:.2f}")

print("\n=== KARŞILAŞTIRMA ===")
print("Rastgele ajan: ~-93 (10 turda)")
print(f"PPO ajan: yukarıdaki sonuçlar")