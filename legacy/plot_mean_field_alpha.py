import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.style.use('prl')

dfname = "run_11_output.h5"
df = h5py.File(dfname,"r")
x = df['x_grid'][:,:,:]
u = df['u'][:,:,:]
v = df['v'][:,:,:]
w = df['w'][:,:,:]
Bx = df['Bx'][:,:,:]
By = df['By'][:,:,:]
Bz = df['Bz'][:,:,:]
cheby_factor = np.sqrt(np.pi**2 - x**2)

u_mean = np.sqrt(np.mean((u**2 + v**2 + w**2)*cheby_factor)/np.mean(cheby_factor))
B_mean = np.sqrt(np.mean((Bx**2 + By**2 + Bz**2)*cheby_factor)/np.mean(cheby_factor))
norm = u_mean*B_mean
alpha = (u*By - v*Bx)
alpha_yz = alpha.mean(axis=0).mean(axis=0)/norm

plt.plot(x[0,0,:], alpha_yz)
plt.xlabel(r"$x$")
plt.ylabel(r"$\langle v_x b_y - v_y b_x \rangle/|v||b|$")

ax =plt.gca()
ax.set_xlim(-np.pi/2,np.pi/2)
ax.set_xticks([-np.pi/2, 0, np.pi/2])
ax.set_xticklabels([r'$-\pi/2$', 0, r'$\pi/2$'])
plt.tight_layout()
plt.savefig('plots/mean_field_alpha_{}.pdf'.format(dfname))

