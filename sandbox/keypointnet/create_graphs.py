import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science'])
plt.rcParams['text.usetex'] = True

plt.rc('xtick', labelsize=10)

"""
data = np.loadtxt("sandbox/keypointnet/results_18.txt")

print(np.mean(data[:, 0] * 1000) / 32, np.std(data[:, 0] * 1000) / 32)

plt.boxplot([data[:, 0], data[:, 1], data[:, 2], data[:, 3]])
plt.ylabel("Weighted Loss")
plt.xticks([0, 1, 2, 3, 4], ["", r'$\mathcal{L}_{mvc}$', r'$\mathcal{L}_{pose}$', r'$\mathcal{L}_{sep}$', r'$\mathcal{L}_{obj}$'])
plt.savefig("sandbox/keypointnet/18.png")
"""

data = np.loadtxt("sandbox/keypointnet/result.txt")
print(data)

mean = np.mean(data, axis=0)
high = np.max(data, axis=0)
low = np.min(data, axis=0)


plt.plot(high)
plt.plot(low)
plt.fill_between(np.arange(data.shape[-1]), high, low, facecolor='C0', alpha=0.4)

plt.ylabel(r'$\dfrac{1}{N} \displaystyle \sum^N_{i=1}D(p_i, q^*_i) \leq k$')

plt.xlabel(r'$k$')
plt.savefig("AUC.png")
