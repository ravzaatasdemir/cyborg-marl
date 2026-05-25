from CybORG import CybORG
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers import PettingZooParallelWrapper
import warnings
import random
warnings.filterwarnings('ignore')

# Ortamı oluştur
sg = DroneSwarmScenarioGenerator()
cyborg = CybORG(sg, 'sim')

# Wrapper — gözlemi sayıya çeviren çevirmen
env = PettingZooParallelWrapper(env=cyborg)

# Oyunu başlat
observations = env.reset()

print("=== OYUN BASLADI ===")
print(f"Toplam ajan sayisi: {len(env.agents)}")
print(f"Mavi ajanlar: {env.agents}")
print()

# blue_agent_0'in ilk gozlemi
obs = observations['blue_agent_0']
print(f"blue_agent_0'in gordugu dunya (ilk 20 sayi):")
print(obs[:20])
print(f"Toplam {len(obs)} sayi var")
print()

# 10 tur oyna
total_rewards = {agent: 0 for agent in env.agents}

for step in range(10):
    # Her ajan icin rastgele aksiyon sec
    actions = {}
    for agent in env.agents:
        action_space = env.action_space(agent)
        actions[agent] = action_space.sample()

    # Aksiyonlari uygula
    observations, rewards, terminations, infos = env.step(actions)

    # Odulleri topla
    for agent in rewards:
        total_rewards[agent] += rewards[agent]

    print(f"Tur {step+1}: blue_agent_0 odulu = {rewards.get('blue_agent_0', 0):.1f}")

print()
print("=== 10 TUR SONU ===")
for agent, reward in total_rewards.items():
    print(f"{agent}: toplam = {reward:.1f}")