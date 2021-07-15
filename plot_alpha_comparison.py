"""
alpha = arctan(ky/kz)

plot three sets of test runs:

* Ng = 100; Nx = 128
* Ng = 100; Nx = 256
* Ng = 200; Nx = 128

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

basedir = Path("data/")

ng100_nx128 = {}
ng100_nx128['runs'] = [1, 2, 4, 3]
ng100_nx128['R'] = [1.01, 1.2, np.sqrt(2.5), 2.0]
ng100_nx128['color'] = 'k'
ng100_nx128['marker'] = 'o'
ng100_nx128['name'] = '$N_g = 100; N_x = 128$'
ng100_nx256 = {}
ng100_nx256['runs'] = [5, 6, 7, 8]
ng100_nx256['R'] = [1.01, 1.2, np.sqrt(2.0), np.sqrt(2.5)]
ng100_nx256['color'] = 'r'
ng100_nx256['marker'] = '*'
ng100_nx256['name'] = '$N_g = 100; N_x = 256$'
ng200_nx128 = {}
ng200_nx128['runs'] = [11, 12, 13, 19, 18]
ng200_nx128['R'] = [1.01, 1.2, np.sqrt(2.0), np.sqrt(2.5), 2.0]
ng200_nx128['color'] = 'b'
ng200_nx128['marker'] = '^'
ng200_nx128['name'] = '$N_g = 200; N_x = 128$'

sets = [ng100_nx128, ng100_nx256, ng200_nx128]



for s in sets:
    runs = s['runs']
    R = s['R']
    ssc = np.array(R)**2
    alpha = []
    for run in runs:
        dfname = Path("run_{:d}_output.h5".format(run))
        df = h5py.File(basedir/dfname, "r")
        #alpha.append(np.arctan2(df['gamma'].attrs['max ky'], df['gamma'].attrs['max kz']))
        kk = np.sqrt(df['gamma'].attrs['max ky']**2 + df['gamma'].attrs['max kz']**2)
        alpha.append(np.arcsin(df['gamma'].attrs['max ky']/kk))
        df.close()
    
    plt.scatter(ssc, alpha, color=s['color'],marker=s['marker'], label=s['name'])

plt.xlabel(r"$S/S_c$")
plt.ylabel(r"$\alpha$")
plt.legend(loc='upper right').draw_frame(False)
plt.savefig("plots/alpha_vs_ssc_comparison.png")
