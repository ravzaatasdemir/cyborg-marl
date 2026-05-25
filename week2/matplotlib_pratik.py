import numpy as np
import matplotlib.pyplot as plt

# 1. Basit çizgi grafik
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 1, 8, 5])

plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', marker='o', label='Veri')
plt.title('Basit Çizgi Grafik')
plt.xlabel('X ekseni')
plt.ylabel('Y ekseni')
plt.legend()
plt.savefig('grafik1.png')
plt.close()
print("grafik1.png kaydedildi")

# 2. Öğrenme eğrisi simülasyonu
episodes = np.arange(1, 101)
rewards = -500 + episodes * 3 + np.random.normal(0, 30, 100)

plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, color='red', alpha=0.5, label='Ham ödül')
smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
plt.plot(smoothed, color='darkred', linewidth=2, label='Ortalama')
plt.title('CybORG Öğrenme Eğrisi (Simülasyon)')
plt.xlabel('Episode')
plt.ylabel('Ödül')
plt.legend()
plt.savefig('grafik2.png')
plt.close()
print("grafik2.png kaydedildi")

# 3. Bar grafik — karşılaştırma
ajanlar = ['Rastgele', 'PPO 50k', 'PPO 200k', 'PPO 500k']
skorlar = [-545, -662, -400, -150]
renkler = ['gray', 'orange', 'blue', 'green']

plt.figure(figsize=(10, 6))
plt.bar(ajanlar, skorlar, color=renkler)
plt.title('Ajan Karşılaştırması')
plt.xlabel('Ajan')
plt.ylabel('Ortalama Ödül')
plt.savefig('grafik3.png')
plt.close()
print("grafik3.png kaydedildi")
