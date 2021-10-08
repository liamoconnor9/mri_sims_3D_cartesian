"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import pickle
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import dedalus.public as de
from dedalus.extras import plot_tools
import publication_settings
import scipy.stats
from configparser import ConfigParser
from pathlib import Path

matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
plt.gcf().subplots_adjust(left=0.15)

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    Nx = 64
    Ny = 256
    Nz = 256
    Lx = np.pi
    ar = 8
    x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
    y_basis = de.Fourier('y', Ny, interval=(0, Lx * ar))
    z_basis = de.Fourier('z', Nz, interval=(0, Lx * ar))
    domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64)
    ky, kz, kx = [domain.elements(i).squeeze() for i in range(3)]
    vx = domain.new_field()
    vy = domain.new_field()
    vz = domain.new_field()

    ky1, kz1 = 0.5, 0.25

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        ke_data = pickle.load(open(path + '/ke_modes.pick', 'rb'))
        ke_ar = ke_data['ke_ar']
        # ke_max_ar = ke_data['ke_max_ar']
        sim_times_ar = ke_data['sim_times_ar']
        for index in range(start, start+count):
            vx['g'] = file['tasks']['vx'][index]
            coeffs1 = vx['c'][ky == ky1, kz == kz1, :]
            coeffs2 = vx['c'][ky == ky1, kz == -kz1, :]
            power_coeff = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))
            ke_ar.append(power_coeff)
            sim_times_ar.append(file['scales/sim_time'][index])
        ke_data['ke_ar'] = ke_ar
        ke_data['sim_times_ar'] = sim_times_ar
        pickle.dump(ke_data, open(path + '/ke_modes.pick', 'wb'))


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync
    import logging
    logger = logging.getLogger(__name__)

    global path 
    path = os.path.dirname(os.path.abspath(__file__))
    write_data = True
    plot = True
    regress = True
    if (write_data):
        args = docopt(__doc__)
        output_path = pathlib.Path('').absolute()
        ke_data = {'ke_ar' : [], 'sim_times_ar' : []}
        pickle.dump(ke_data, open(path + '/ke_modes.pick', 'wb'))

        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit_writes(args['<files>'], main, output=output_path)

    if (plot):
        fig = plt.figure()
        write_suffix = args['--suffix']
        ke_data = pickle.load(open(path + '/ke_modes.pick', 'rb'))
        ke_ar = ke_data['ke_ar']
        sim_times_ar = ke_data['sim_times_ar']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.plot(sim_times_ar, ke_ar, color=colors[-1], label='Simulation')
        plt.yscale('log')

        plt.xlabel(r'$t$')
        plt.ylabel(r'$\langle |\mathbf{u}|^2 \rangle_{\mathcal{D}}$')
        plt.title(r'$\rm{Re}^{-1} \, = \, \eta \, = \nu \, = \, 10^{-3}; \; S/S_C = 1.2$')
        plt.savefig(path + '/ke_modes_' + write_suffix + '.png')