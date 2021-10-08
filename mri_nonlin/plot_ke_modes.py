"""
Plot planes from joint analysis files.

Usage:
    plot_ke_modes.py <files>... [--suffix=write_string_suffix] 

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

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        ke_data = pickle.load(open(path + '/ke_modes.pick', 'rb'))
        ke_ar1 = ke_data['ke_ar1']
        ke_ar2 = ke_data['ke_ar2']
        # ke_max_ar = ke_data['ke_max_ar']
        sim_times_ar = ke_data['sim_times_ar']
        for index in range(start, start+count):
            field['g'] = file['tasks']['vx'][index]
            coeffs1 = field['c'][ky == ky1, kz == kz1, :]
            # coeffs2 = field['c'][-ky == ky1, kz == kz1, :]
            # power_coeff_x1 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))
            power_coeff_x1 = np.sum(coeffs1*np.conj(coeffs1))
            
            coeffs1 = field['c'][ky == ky2, kz == kz2, :]
            coeffs2 = field['c'][-ky == ky2, kz == kz2, :]
            power_coeff_x2 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))
            
            field['g'] = file['tasks']['vy'][index]
            coeffs1 = field['c'][ky == ky1, kz == kz1, :]
            # coeffs2 = field['c'][-ky == ky1, kz == kz1, :]
            # power_coeff_y1 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))
            power_coeff_y1 = np.sum(coeffs1*np.conj(coeffs1))

            coeffs1 = field['c'][ky == ky2, kz == kz2, :]
            coeffs2 = field['c'][-ky == ky2, kz == kz2, :]
            power_coeff_y2 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))
            
            field['g'] = file['tasks']['vz'][index]
            coeffs1 = field['c'][ky == ky1, kz == kz1, :]
            # coeffs2 = field['c'][-ky == ky1, kz == kz1, :]
            # power_coeff_z1 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))
            power_coeff_z1 = np.sum(coeffs1*np.conj(coeffs1))

            coeffs1 = field['c'][ky == ky2, kz == kz2, :]
            coeffs2 = field['c'][-ky == ky2, kz == kz2, :]
            power_coeff_z2 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))

            ke_ar1.append(power_coeff_x1 + power_coeff_y1 + power_coeff_z1)
            ke_ar2.append(power_coeff_x2 + power_coeff_y2 + power_coeff_z2)
            sim_times_ar.append(file['scales/sim_time'][index])
        ke_data['ke_ar1'] = ke_ar1
        ke_data['ke_ar2'] = ke_ar2
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
    args = docopt(__doc__)

    global ky1, ky2, kz1, kz2
    ky1, kz1 = 0.0, 0.5
    ky2, kz2 = 0.25, 0.5  

    if (write_data):
        output_path = pathlib.Path('').absolute()
        ke_data = {'ke_ar1' : [], 'ke_ar2' : [], 'sim_times_ar' : []}
        pickle.dump(ke_data, open(path + '/ke_modes.pick', 'wb'))
   
        Nx = 64
        Ny = 256
        Nz = 256
        Lx = np.pi
        ar = 8
        x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
        y_basis = de.Fourier('y', Ny, interval=(0, Lx * ar))
        z_basis = de.Fourier('z', Nz, interval=(0, Lx * ar))
        domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64)
        
        global ky, kz, kx
        ky, kz, kx = [domain.elements(i).squeeze() for i in range(3)]
        
        global field
        field = domain.new_field() 
        
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit_writes(args['<files>'], main, output=output_path)

    if (plot):
        fig = plt.figure()
        write_suffix = args['--suffix']
        diff_str_dict = {'diff1en2' : '10^{-2}', 'diff1en3' : '10^{-3}'}
        diff_str = diff_str_dict[write_suffix[:8]]
        ke_data = pickle.load(open(path + '/ke_modes.pick', 'rb'))
        ke_ar1 = ke_data['ke_ar1']
        ke_ar2 = ke_data['ke_ar2']
        sim_times_ar = ke_data['sim_times_ar']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.yscale('log')
        if (regress):
            ln_be_ar = np.log(ke_ar1)
            t_cutoff = 25
            i_cutoff = -1
            for i, t in enumerate(sim_times_ar):
                if (t > t_cutoff):
                    i_cutoff = i
                    break
            t_cutoff2 = 130
            i_cutoff2 = -1
            for i, t in enumerate(sim_times_ar):
                if (t > t_cutoff2):
                    i_cutoff2 = i
                    break            
            t_lin = sim_times_ar[i_cutoff:i_cutoff2]
            ke_lin = ln_be_ar[i_cutoff:i_cutoff2]
            slope1, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_lin, ke_lin)
            print('R2 val: ' + str(r_value**2))
            print('slope: ' + str(slope1))
            print('intercept: ' + str(intercept))      
            ke_reg = np.exp(slope1 * np.array(t_lin) + intercept)
            plt.plot(t_lin, ke_reg, color='k', linestyle='--', linewidth=2)

            # fig.text(.2, .05, r'$k_y, k_z \, = \, $' + str(ky1) + ',' + str(kz1) + ': Growth rate = ' + str(round(slope, 4)), ha='center')

            ln_be_ar = np.log(ke_ar2)
            t_cutoff = 25
            i_cutoff = -1
            for i, t in enumerate(sim_times_ar):
                if (t > t_cutoff):
                    i_cutoff = i
                    break
            t_cutoff2 = 130
            i_cutoff2 = -1
            for i, t in enumerate(sim_times_ar):
                if (t > t_cutoff2):
                    i_cutoff2 = i
                    break            
            t_lin = sim_times_ar[i_cutoff:i_cutoff2]
            ke_lin = ln_be_ar[i_cutoff:i_cutoff2]
            slope2, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_lin, ke_lin)
            print('R2 val: ' + str(r_value**2))
            print('slope: ' + str(slope2))
            print('intercept: ' + str(intercept))      
            ke_reg = np.exp(slope2 * np.array(t_lin) + intercept)
            plt.plot(t_lin, ke_reg, color='k', linestyle='--', linewidth=2)

            # fig.text(.2, -.05, r'$k_y, k_z \, = \, $' + str(ky2) + ',' + str(kz2) + ': Growth rate = ' + str(round(slope, 4)), ha='center')
            plt.plot(sim_times_ar, ke_ar1, label=r'$k_y, k_z \, = \, $' + str(ky1) + ',' + str(kz1) + '; growth rate = ' + str(round(slope1.real, 4)))
            plt.plot(sim_times_ar, ke_ar2, label=r'$k_y, k_z \, = \, $' + str(ky2) + ',' + str(kz2) + '; growth rate = ' + str(round(slope2.real, 4)))

            plt.legend()        # plt.xlim(0, 400)


        plt.xlabel(r'$t$')
        plt.ylabel(r'$\langle |\mathbf{u}|^2 \rangle_{\mathcal{D}}$')
        plt.legend()
        plt.title(r'$\rm{Re}^{-1} \, = \, \eta \, = \nu \, = \, ' + diff_str + '; \; S/S_C = 1.02$')
        plt.savefig(path + '/ke_modes_' + write_suffix + '.png')
