"""
Plot kinetic and magnetic energies from joint analysis files.

Usage:
    plot_kebe.py <files>... [--dir=<dir>, --config=<config_file>, --suffix=<run_suffix>]

"""

from docopt import docopt
from configparser import ConfigParser
import h5py
import pickle
import os
import sys
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
# from dedalus.extras import plot_tools
import publication_settings
import logging
logger = logging.getLogger(__name__)
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
plt.gcf().subplots_adjust(left=0.15)

def title_string(write_suffix):
    title_str = (r'$\nu \, = \,' + str(nu) + ', \, \eta \, = \, $' + str(eta) + r'$; \; S/S_C = $' + str(SdivbySc))
    tau = config.getfloat('parameters','tau')
    if (tau != 0):
        title_str += r'$; \, \tau = $' + str(tau)
    if (config.getboolean('parameters','isNoSlip')):
        title_str += '; No Slip BCs'
    else:
        title_str += '; Stress Free BCs'
    return title_str

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        be_data = pickle.load(open(path + '/kebe_data_' + write_suffix + '.pick', 'rb'))
        ke_ar = be_data['ke_ar']
        be_ar = be_data['be_ar']
        # ke_max_ar = be_data['ke_max_ar']
        sim_times_ar = be_data['sim_times_ar']
        for index in range(start, start+count):
            ke_avg = file['tasks']['ke_tot'][index][0, 0, 0]
            be_avg = file['tasks']['be'][index][0, 0, 0]
            ke_ar.append(ke_avg)
            be_ar.append(be_avg)
            sim_times_ar.append(file['scales/sim_time'][index])
        be_data['ke_ar'] = ke_ar
        be_data['be_ar'] = be_ar
        be_data['sim_times_ar'] = sim_times_ar
        pickle.dump(be_data, open(path + '/kebe_data_' + write_suffix + '.pick', 'wb'))
        


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    global path 
    global write_suffix
    args = docopt(__doc__)
    dir = args['--dir']
    write_suffix = args['--suffix']
    filename = dir + write_suffix + '/' + args['--config']

    config = ConfigParser()
    config.read(str(filename))
    path = os.path.dirname(os.path.abspath(__file__))
    write_data = True
    plot = True
    regress = False

    if (write_data):
        output_path = pathlib.Path('').absolute()
        be_data = {'be_ar' : [], 'ke_ar' : [], 'be_perts_ar' : [], 'ke_perts_ar' : [], 'sim_times_ar' : []}
        pickle.dump(be_data, open(path + '/kebe_data_' + write_suffix + '.pick', 'wb'))

        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit_writes(args['<files>'], main, output=output_path)

    if (plot):
        fig = plt.figure()
        nu = config.getfloat('parameters','nu')
        Pm = config.getfloat('parameters','Pm')
        eta = nu / Pm
        R = config.getfloat('parameters','R')

        be_data = pickle.load(open(path + '/kebe_data_' + write_suffix + '.pick', 'rb'))
        ke_ar = be_data['ke_ar']
        be_ar = be_data['be_ar']
        sim_times_ar = be_data['sim_times_ar']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        shave_ind = 10
        plt.plot(sim_times_ar[:-shave_ind], np.array(ke_ar[:-shave_ind]), color='darkblue', label=r'$\langle |\mathbf{u}|^2 \rangle_{\mathcal{D}}$')
        plt.plot(sim_times_ar[:-shave_ind], np.array(be_ar[:-shave_ind]), color='darkred', label=r'$\langle |\mathbf{b}|^2 \rangle_{\mathcal{D}}$')
        plt.yscale('log')
        if (regress):
            ln_be_ar = np.log(be_ar)
            t_cutoff = 40
            i_cutoff = -1
            for i, t in enumerate(sim_times_ar):
                if (t > t_cutoff):
                    i_cutoff = i
                    break
            t_cutoff2 = 90
            i_cutoff2 = -1
            for i, t in enumerate(sim_times_ar):
                if (t > t_cutoff2):
                    i_cutoff2 = i
                    break            
            t_lin = sim_times_ar[i_cutoff:i_cutoff2]
            ke_lin = ln_be_ar[i_cutoff:i_cutoff2]
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_lin, ke_lin)
            print('R2 val: ' + str(r_value**2))
            print('slope: ' + str(slope))
            print('intercept: ' + str(intercept))      
            ke_reg = np.exp(slope * np.array(t_lin) + intercept)
            plt.plot(t_lin, ke_reg, color='k', linestyle='--', linewidth=2, label='Best fit')

            fig.text(.2, .05, 'Growth rate = ' + str(round(slope, 4)), ha='center')

        q = 0.75
        S      = -R*np.sqrt(q)
        f      =  R/np.sqrt(q)
        SdivbySc = S / -1.0 * f
        plt.xlabel(r'$t$')
        plt.ylabel('Average Energy')

        plt.title(title_string(write_suffix))
        plt.legend()        # plt.xlim(0, 400)
        plt.savefig(dir + write_suffix + '/kebe_tot_nonlin_' + write_suffix + '.png')
