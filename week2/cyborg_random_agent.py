from CybORG import CybORG
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator
import warnings
import random
warnings.filterwarnings('ignore')

# Ortamı oluştur
sg = DroneSwarmScenarioGenerator()
cyborg = CybORG(sg, 'sim')

episodes = 5
steps_per_episode = 50

for episode in range(episodes):
    result = cyborg.reset()
    total_reward = 0

    for step in range(steps_per_episode):
        # Aksiyon uzayını al ve rastgele seç
        action_space = cyborg.get_action_space('blue_agent_0')
        actions = list(action_space.keys())
        action = random.choice(actions)

        result = cyborg.step(action=action, agent='blue_agent_0')
        total_reward += result.reward

    print(f"Episode {episode + 1}: Toplam ödül = {total_reward:.2f}")

print("\nBitti!")