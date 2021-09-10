"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--suffix=write_string_suffix] 

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
import scipy.stats

matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
plt.gcf().subplots_adjust(left=0.15)

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        ke_data = pickle.load(open(path + '/ke_modes_data_mri2.pick', 'rb'))
        ke_ar_00 = []
        ke_ar_10 = []
        ke_ar_01 = []
        ke_ar_11 = []
        # ke_max_ar = ke_data['ke_max_ar']
        sim_times_ar = []
        ke_2d = file['tasks']['ke_2D']
        times = file['scales/sim_time']
        for index in range(start, start+count):
            ke_ar_00.append(ke_2d[index][0, 0, 0])
            # ke_ar_10.append(abs(ke_2d[index][1, 0, 0]))
            # ke_ar_01.append(abs(ke_2d[index][0, 1, 0]))
            ke_ar_10.append(abs(ke_2d[index][1, 0, 0] + ke_2d[index][-1, 0, 0]))
            ke_ar_01.append(abs(ke_2d[index][0, 1, 0]))
            ke_ar_11.append(abs(ke_2d[index][1, 1, 0] + ke_2d[index][-1, 1, 0]))
            sim_times_ar.append(times[index])
        ke_data['ke_ar_00'] += ke_ar_00
        ke_data['ke_ar_10'] += ke_ar_10
        ke_data['ke_ar_01'] += ke_ar_01
        ke_data['ke_ar_11'] += ke_ar_11
        ke_data['sim_times_ar'] += sim_times_ar
        pickle.dump(ke_data, open(path + '/ke_modes_data_mri2.pick', 'wb'))
        


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    global path 
    path = os.path.dirname(os.path.abspath(__file__))
    write_data = False
    plot = True
    regress = True
    args = docopt(__doc__)
    if (write_data):
        output_path = pathlib.Path('').absolute()
        ke_data = {'ke_ar_00' : [], 'ke_ar_01' : [], 'ke_ar_10' : [], 'ke_ar_11' : [], 'sim_times_ar' : []}
        pickle.dump(ke_data, open(path + '/ke_modes_data_mri2.pick', 'wb'))

        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit_writes(args['<files>'], main, output=output_path)
    if (plot):
        fig = plt.figure()
        write_suffix = args['--suffix']
        ke_data = pickle.load(open(path + '/ke_modes_data_mri2.pick', 'rb'))
        ke_ar_00 = ke_data['ke_ar_00']
        ke_ar_10 = ke_data['ke_ar_10']
        ke_ar_01 = ke_data['ke_ar_01']
        ke_ar_11 = ke_data['ke_ar_11']
        sim_times_ar = ke_data['sim_times_ar']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cutoff = -100
        plt.plot(sim_times_ar[:cutoff], ke_ar_00[:cutoff], label='(ky, kz) = (0, 0)')
        plt.plot(sim_times_ar[:cutoff], ke_ar_10[:cutoff], label='(ky, kz) = (0.25, 0)')
        plt.plot(sim_times_ar[:cutoff], ke_ar_01[:cutoff], label='(ky, kz) = (0, 0.25)')
        plt.plot(sim_times_ar[:cutoff], ke_ar_11[:cutoff], label='(ky, kz) = (0.25, 0.25)')
        plt.yscale('log')
        # if (regress):
        #     ln_ke_ar = np.log(ke_ar)
        #     t_cutoff = 25
        #     i_cutoff = -1
        #     for i, t in enumerate(sim_times_ar):
        #         if (t > t_cutoff):
        #             i_cutoff = i
        #             break
        #     t_cutoff2 = 60
        #     i_cutoff2 = -1
        #     for i, t in enumerate(sim_times_ar):
        #         if (t > t_cutoff2):
        #             i_cutoff2 = i
        #             break
            
        #     t_lin = sim_times_ar[i_cutoff:i_cutoff2]
        #     ke_lin = ln_ke_ar[i_cutoff:i_cutoff2]
        #     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_lin, ke_lin)
        #     print('R2 val: ' + str(r_value**2))
        #     print('slope: ' + str(slope))
        #     print('intercept: ' + str(intercept))      
        #     ke_reg = np.exp(slope * np.array(t_lin) + intercept)
        #     plt.plot(t_lin, ke_reg, color='k', linestyle='--', linewidth=2, label='Best fit')

        #     fig.text(.2, .05, 'Growth rate = ' + str(round(slope, 4)), ha='center')
        #     plt.legend()

        # plt.xlim(0, 50)
        plt.xlabel(r'$t$')
        plt.ylabel(r'KE mode amplitude')
        plt.legend()
        plt.title(r'$\rm{Re}^{-1} \, = \, \eta \, = \nu \, = \, 10^{-3}; \; S/S_C = 1.2$')
        plt.savefig(path + '/ke_modes3_nonlin_' + write_suffix + '.png')
