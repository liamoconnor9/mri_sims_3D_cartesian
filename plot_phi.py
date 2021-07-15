"""
phi = arctan(ky/kz)

"""

import numpy as np
import numpy.polynomial.polynomial as poly
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from basic_units import radians

plt.style.use('prl')

use_opt = False

def calc_phi():
    basedir = Path("data/")
    runs = [34, 33, 31, 21, 15, 11, 12, 16, 30, 13, 27, 28, 29, 26, 24, 17, 14, 18]
    R = [np.sqrt(0.2), np.sqrt(0.3), np.sqrt(0.4), np.sqrt(0.5), 0.8, 1.01, 1.2, np.sqrt(1.75), 1.375, np.sqrt(2.0), np.sqrt(2.01), np.sqrt(2.015), 1.425, np.sqrt(2.05), np.sqrt(2.1), 1.5, np.sqrt(2.5), 2.0]
    ssc = np.array(R)**2
    phi = []

    growth = []
    dk = []
    for run in runs:
        dfname = Path("run_{:d}_output.h5".format(run))
        df = h5py.File(basedir/dfname, "r")
        if use_opt:
            #phi.append(np.arctan2(df['gamma'].attrs['max ky'], df['gamma'].attrs['max kz']))
            ky = df['gamma'].attrs['max ky']
            kx = df['gamma'].attrs['max kz']
            gamma_max = df['gamma'].attrs['max growth rate']
        else:
            gamma_global = df['gamma'][:,:]
            max_where = np.unravel_index(gamma_global.argmax(), gamma_global.shape)
            ky = df['ky'][max_where[0]] 
            kz = df['kz'][max_where[1]]
            gamma_max = gamma_global.real.max()
        dk.append(df['ky'][1]-df['ky'][0])
        kk = np.sqrt(ky**2 + kz**2)
        phi.append(np.arcsin(ky/kk))
        growth.append(gamma_max)
        df.close()
    phi = np.array(phi)
    ssc = np.array(ssc)
    dk = np.array(dk)

    return phi, ssc, dk, growth

if __name__ == "__main__":
    phi, ssc, dk, growth = calc_phi()
    fig, ax = plt.subplots()
    c = ax.scatter(ssc, phi, marker='o', c=growth, yunits=radians, zorder=2)#,edgecolor='k',linewidth=0.8)
    fig.colorbar(c, ax = ax, label='$\gamma/S$')
    ax.set_xlabel(r"$S/S_c$")
    ax.set_ylabel(r"$\phi$")
    ax.set_yticklabels([0, "$\pi/8$", "$\pi/4$"])
    ax.set_yticks([0, np.pi/8,np.pi/4])
    ssc_new = np.linspace(ssc[0], ssc[-1], len(ssc)*10)
    a_thresh = 0.01
    coefs = poly.polyfit(ssc[phi > a_thresh], phi[phi > a_thresh], 1)
    ffit = poly.polyval(ssc_new, coefs)
    #ax.plot(ssc_new, ffit)

    axins = ax.inset_axes([0.7, 0.7, 0.24, 0.24])
    axins.scatter(ssc, phi, marker='o',c=growth, yunits=radians)
    # sub region of the original image
    x1, x2, y1, y2 = 1.4**2, 1.45**2, -0.05, 0.15
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticklabels('')
    #axins.set_yticklabels('')

    ax.axhline(0,color='k', alpha=0.4,zorder=1)
    ax.axvspan(-1,0.102,color='k', alpha=0.4)
    ax.set_ylim(-0.05,1.0)
    ax.set_xlim(-0.5,4.1)
    fig.tight_layout()
    if use_opt:
        fig.savefig("plots/phi_vs_ssc.pdf")
    else:
        fig.savefig("plots/phi_vs_ssc_grid.pdf")

    # plot against R
    ax.clear()
    ax.scatter(R, phi, marker='o',c=growth, yunits=radians)

    axins = ax.inset_axes([0.7, 0.7, 0.24, 0.24])
    axins.scatter(R, phi, marker='o',c=growth, yunits=radians)
    # sub region of the original image
    x1, x2, y1, y2 = 1.4, 1.45, -0.05, 0.15
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticklabels('')
    #axins.set_yticklabels('')

    R_new = np.sqrt(ssc_new)
    ax.plot(R_new, ffit)
    #ax.colorbar(label='$\gamma$')
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$\phi$")
    ax.axhline(0,color='k', alpha=0.4)
    ax.set_ylim(-0.1,1.0)
    if use_opt:
        fig.savefig("plots/phi_vs_R.pdf")
    else:
        fig.savefig("plots/phi_vs_R_grid.pdf")


    # plot against (S/Sc)^2
    ax.clear()
    ssc2 = ssc**2
    ax.scatter(ssc2, phi, marker='o',c=growth, yunits=radians)

    axins = ax.inset_axes([0.7, 0.7, 0.24, 0.24])
    axins.scatter(ssc2, phi, marker='o',c=growth, yunits=radians)
    # sub region of the original image
    x1, x2, y1, y2 = 1.4, 1.45, -0.05, 0.15
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.set_xticklabels('')
    #axins.set_yticklabels('')

    #ssc2_new = np.sqrt(ssc_new)
    #ax.plot(ssc2_new, ffit)
    #ax.colorbar(label='$\gamma$')
    ax.set_xlabel(r"$(S/S_c)^2$")
    ax.set_ylabel(r"$\phi$")
    ax.axhline(0,color='k', alpha=0.4)
    ax.set_ylim(-0.1,1.0)
    if use_opt:
        fig.savefig("plots/phi_vs_ssc2.pdf")
    else:
        fig.savefig("plots/phi_vs_ssc2_grid.pdf")

