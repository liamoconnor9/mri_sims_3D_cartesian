import numpy as np
import matplotlib.pyplot as plt
import os

data = np.loadtxt('vp_growth_rates_By_svd.txt', skiprows=1, delimiter=', ')
Bs = data[:, 0]
evs = data[:, 3]

path = os.path.dirname(os.path.abspath(__file__))
plt.scatter(Bs, evs)
plt.title('Max Growth Rates')
plt.xlabel(r'$\langle B_y \rangle_{\rm{EVP}} \; / \; \langle B_y \rangle_{\rm{Sim}}$')
plt.ylabel('Growth Rate')
plt.savefig(path + '/max_evs.png')

# print(data.shape)