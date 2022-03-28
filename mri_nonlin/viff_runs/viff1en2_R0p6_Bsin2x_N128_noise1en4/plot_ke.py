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
# from dedalus.extras import plot_tools
import publication_settings
import scipy.stats

matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
plt.gcf().subplots_adjust(left=0.15)

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        ke_data = pickle.load(open(path + '/ke_data_' + write_suffix + '.pick', 'rb'))
        ke_ar = ke_data['ke_ar']
        # ke_max_ar = ke_data['ke_max_ar']
        sim_times_ar = ke_data['sim_times_ar']
        for index in range(start, start+count):
            ke_avg = file['tasks']['ke'][index][0, 0, 0]
            ke_ar.append(ke_avg)
            sim_times_ar.append(file['scales/sim_time'][index])
        ke_data['ke_ar'] = ke_ar
        ke_data['sim_times_ar'] = sim_times_ar
        pickle.dump(ke_data, open(path + '/ke_data_' + write_suffix + '.pick', 'wb'))
        


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    global path 
    global write_suffix
    args = docopt(__doc__)
    write_suffix = args['--suffix']
    path = os.path.dirname(os.path.abspath(__file__))
    write_data = True
    plot = True
    regress = True
    if (write_data):
        output_path = pathlib.Path('').absolute()
        ke_data = {'ke_ar' : [], 'sim_times_ar' : []}
        pickle.dump(ke_data, open(path + '/ke_data_' + write_suffix + '.pick', 'wb'))

        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit_writes(args['<files>'], main, output=output_path)
    if (plot):
        fig = plt.figure()
        diff_str_dict = {'diff1en2' : '$10^{-2}$', 'viff1en2' : '$10^{-2}$', 'viff2en3' : r'$2 \times 10^{-3}$', 'viff1en3' : '$10^{-3}$', 'diff2en3' : r'$2 \times 10^{-3}$', 'diff3en3' : r'$3 \times 10^{-3}$', 'diff1en3' : '10^{-3}'}
        # diff_str_dict = {'diff1en2' : '10^{-2}', 'diff2en3' : r'$2 \times 10^{-3}$', 'diff3en3' : r'$3 \times 10^{-3}$', 'diff1en3' : '10^{-3}'}
        diff_str = diff_str_dict[write_suffix[:8]]
        ke_data = pickle.load(open(path + '/ke_data_' + write_suffix + '.pick', 'rb'))
        ke_ar = ke_data['ke_ar']
        sim_times_ar = ke_data['sim_times_ar']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        shave_ind = 10
        plt.plot(sim_times_ar[:-shave_ind], ke_ar[:-shave_ind], color=colors[-1], label='Simulation')
        plt.yscale('log')
        if (regress):
            ln_ke_ar = np.log(ke_ar)
            t_cutoff = 120
            i_cutoff = -1
            for i, t in enumerate(sim_times_ar):
                if (t > t_cutoff):
                    i_cutoff = i
                    break
            t_cutoff2 = 160
            i_cutoff2 = -1
            for i, t in enumerate(sim_times_ar):
                if (t > t_cutoff2):
                    i_cutoff2 = i
                    break
            
            t_lin = sim_times_ar[i_cutoff:i_cutoff2]
            ke_lin = ln_ke_ar[i_cutoff:i_cutoff2]
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_lin, ke_lin)
            print('R2 val: ' + str(r_value**2))
            print('slope: ' + str(slope))
            print('intercept: ' + str(intercept))      
            ke_reg = np.exp(slope * np.array(t_lin) + intercept)
            plt.plot(t_lin, ke_reg, color='k', linestyle='--', linewidth=2, label='Best fit')

            fig.text(.2, .05, 'Growth rate = ' + str(round(slope, 4)), ha='center')
            plt.legend()

        plt.xlabel(r'$t$')
        plt.ylabel(r'$\langle |\mathbf{u}|^2 \rangle_{\mathcal{D}}$')
        plt.title(r'$\rm{Re}^{-1} \, = \, \eta \, = \nu \, = \, $' + diff_str + r'$; \; S/S_C = 1.02$')
        # plt.title(r'$\rm{Re}^{-1} \, = \, \eta \, = \nu \, = \, ' + diff_str + '; \; S/S_C = 1.02$')
        plt.savefig(path + '/ke_nonlin_' + write_suffix + '.png')
