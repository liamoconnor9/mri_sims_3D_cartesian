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
from dedalus.extras import plot_tools
import publication_settings

matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
plt.gcf().subplots_adjust(left=0.15)

fig = plt.figure()

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        be_data = pickle.load(open(path + '/be_data_mri.pick', 'rb'))
        be_ar = be_data['be_ar']
        # ke_max_ar = be_data['ke_max_ar']
        sim_times_ar = be_data['sim_times_ar']
        for index in range(start, start+count):
            ke_avg = file['tasks']['be'][index][0, 0, 0]
            be_ar.append(ke_avg)
            sim_times_ar.append(file['scales/sim_time'][index])
        be_data['be_ar'] = be_ar
        be_data['sim_times_ar'] = sim_times_ar
        pickle.dump(be_data, open(path + '/be_data_mri.pick', 'wb'))
        


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    global path 
    path = os.path.dirname(os.path.abspath(__file__))
    write_data = True
    plot = True
    if (write_data):
        args = docopt(__doc__)
        output_path = pathlib.Path(args['--output']).absolute()
        be_data = {'be_ar' : [], 'sim_times_ar' : []}
        pickle.dump(be_data, open(path + '/be_data_mri.pick', 'wb'))

        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit_writes(args['<files>'], main, output=output_path)
    if (plot):
        be_data = pickle.load(open(path + '/be_data_mri.pick', 'rb'))
        be_ar = be_data['be_ar']
        sim_times_ar = be_data['sim_times_ar']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.plot(sim_times_ar, be_ar, color=colors[-1])
        plt.yscale('log')
        # plt.xlim(0, 400)
        # plt.legend(frameon=False)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\langle |\mathbf{b}|^2 \rangle_{\mathcal{D}}$')
        plt.title(r'$\rm{Re}^{-1} \, = \, \eta \, = \nu \, = \, 2 \cdot 10^{-4}; \; S/S_C = 1.2$')
        plt.savefig(path + '/be_diff2en4.png')
