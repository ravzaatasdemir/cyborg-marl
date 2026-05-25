from CybORG import CybORG
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers import PettingZooParallelWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Ortamı tek ajanlı hale getiren wrapper
class SingleAgentWrapper(gym.Env):
    def __init__(self):
        sg = DroneSwarmScenarioGenerator()
        cyborg = CybORG(sg, 'sim')
        self.env = PettingZooParallelWrapper(env=cyborg)
        self.agent = 'blue_agent_0'

        obs_space = self.env.observation_space(self.agent)
        act_space = self.env.action_space(self.agent)

        self.observation_space = gym.spaces.Box(
            low=0, high=200,
            shape=obs_space.shape,
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(act_space.n)

    def reset(self, seed=None, options=None):
        observations = self.env.reset()
        obs = observations[self.agent].astype(np.float32)
        return obs, {}

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

# Ortamı oluştur
print("Ortam hazırlanıyor...")
env = DummyVecEnv([SingleAgentWrapper])

# PPO ajanını oluştur
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    n_epochs=5,
    gamma=0.99,
    device='cpu'
)

# Eğit
print("Eğitim başlıyor...")
model.learn(total_timesteps=50000)

# Kaydet
model.save("cyborg_ppo_model")
print("Model kaydedildi: cyborg_ppo_model.zip")
env.close()