import matplotlib.pyplot as plt
import h5py
import sys
import numpy as np
from pathlib import Path

plt.style.use('prl')
filename = Path(sys.argv[-1])

rows = 2
cols = 3
names = [r'$v_x$',r'$v_y$',r'$v_z$',r'$b_x$',r'$b_y$',r'$b_z$']
cmaps = ['RdBu','RdBu','RdBu','PuOr','PuOr','PuOr']
plt.figure(figsize=(12,6))
with h5py.File(filename, 'r') as df:
    x = df['eigvecs'].attrs['x']
    for i in range(1,7):
        plt.subplot(rows,cols,i)
        plt.plot(x, df['eigvecs'][i,:].real, label='real')
        plt.plot(x, df['eigvecs'][i,:].imag, label='imag')
        if i == 1:
            plt.legend()
        plt.xlabel(r'$x$')
        plt.ylabel(names[i-1])

    plt.tight_layout()
    plt.savefig('plots/' + filename.stem + 'eigvecs')

    # 2D plots
    n_yz = 128
    ky = df['eigvals'].attrs['ky']
    kz = df['eigvals'].attrs['kz']
    Ly = 2*np.pi/ky
    Lz = 2*np.pi/kz

    y = np.linspace(0, Ly, n_yz, endpoint=False)

    yyy, xxx = np.meshgrid(y,x,indexing='ij')
    y_dep = np.exp(1j*ky*yyy)
    print(ky)
    plt.figure(figsize=(10,10))
    eigvec = df['eigvecs']
    for i in range(1,7):
        ax = plt.subplot(rows,cols,i)
        pcol = plt.pcolormesh(xxx,yyy,(eigvec[i,:]*y_dep).real,linewidth=0,rasterized=True, cmap=cmaps[i-1])
        #pcol.set_edgecolor('face')
        plt.title(names[i-1])
        if i !=4:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        else:
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            ax.set_xticklabels(["$-\pi/2$", 0, "$\pi/2$"])
            ax.set_xticks([-np.pi/2, 0, np.pi/2])

    plt.savefig('plots/eigvecs_xy_' + filename.stem)
