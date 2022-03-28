"""
Plot planes from joint analysis files.

Usage:
    plot_be_modes.py <files>... [--suffix=write_string_suffix] 

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
        be_data = pickle.load(open(path + '/be_modes.pick', 'rb'))
        be_ar1 = be_data['be_ar1']
        be_ar2 = be_data['be_ar2']
        be_ar3 = be_data['be_ar3']
        # be_max_ar = be_data['be_max_ar']
        sim_times_ar = be_data['sim_times_ar']
        for index in range(start, start+count):
            field1['g'] = file['tasks']['Ax'][index]
            field2['g'] = file['tasks']['Az'][index]
            by = de.operators.differentiate(field1, 'z') - de.operators.differentiate(field2, 'x')
            by['c']

            # by['g'] = file['tasks']['Ay'][index]
            # coeffs1 = by['c'][ky == ky1, kz == kz1, :]
            # # coeffs2 = field['c'][-ky == ky1, kz == kz1, :]
            # # power_coeff_y1 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))
            # power_coeff_y1 = np.sum(coeffs1*np.conj(coeffs1))

            # coeffs1 = by['c'][ky == ky2, kz == kz2, :]
            # coeffs2 = by['c'][-ky == ky2, kz == kz2, :]
            # power_coeff_y2 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))
            
            # field['g'] = file['tasks']['Az'][index]
            # coeffs1 = field['c'][ky == ky1, kz == kz1, :]
            # # coeffs2 = field['c'][-ky == ky1, kz == kz1, :]
            # # power_coeff_z1 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))
            # power_coeff_z1 = np.sum(coeffs1*np.conj(coeffs1))

            # coeffs1 = field['c'][ky == ky2, kz == kz2, :]
            # coeffs2 = field['c'][-ky == ky2, kz == kz2, :]
            # power_coeff_z2 = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))

            be_ar1.append(((by['c'][[ky == 0.0, kz == 0.25, 0]]) * np.conj((by['c'][[ky == 0.0, kz == 0.25, 0]]))).real)
            be_ar2.append(((by['c'][[ky == 0.0, kz == -0.25, 0]]) * np.conj((by['c'][[ky == 0.0, kz == 0.5, 0]]))).real)
            be_ar3.append(((by['c'][[ky == 0.0, kz == 0.5, 0]]) * np.conj((by['c'][[ky == 0.0, kz == 0.75, 0]]))).real)
            # ind = np.where(by['c'] * np.conj(by['c']) == np.amax(by['c'] * np.conj(by['c'])))
            # print(ky[ind[0]])
            # print(kz[ind[1]])
            # print(kx[ind[2]])
            # print(by['c'][ind])
            # # print(be_ar1)
            # sys.exit()
            sim_times_ar.append(file['scales/sim_time'][index])
        be_data['be_ar1'] = be_ar1
        be_data['be_ar2'] = be_ar2
        be_data['be_ar3'] = be_ar3
        be_data['sim_times_ar'] = sim_times_ar
        pickle.dump(be_data, open(path + '/be_modes.pick', 'wb'))


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
    # ky1, kz1 = 0.0, 0.0, 0.0
    # ky2, kz2 = 0.0, 0.25, 0.0

    if (write_data):
        output_path = pathlib.Path('').absolute()
        be_data = {'be_ar1' : [], 'be_ar2' : [], 'be_ar3' : [], 'sim_times_ar' : []}
        pickle.dump(be_data, open(path + '/be_modes.pick', 'wb'))
   
        ar = 8
        Ny = 128
        Nz = 128
        Nx = 32
        Lx = np.pi
        x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
        y_basis = de.Fourier('y', Ny, interval=(0, Lx * ar))
        z_basis = de.Fourier('z', Nz, interval=(0, Lx * ar))
        domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64)
        
        global ky, kz, kx
        ky, kz, kx = [domain.elements(i).squeeze() for i in range(3)]
        
        global field
        field1 = domain.new_field() 
        field2 = domain.new_field() 
        
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
        diff_str = '10^{-2}'
        be_data = pickle.load(open(path + '/be_modes.pick', 'rb'))
        be_ar1 = be_data['be_ar1']
        be_ar2 = be_data['be_ar2']
        be_ar3 = be_data['be_ar3']
        sim_times_ar = be_data['sim_times_ar']
        plt.yscale('log')
        plt.plot(sim_times_ar, be_ar1, label=r'$(n_y, n_z, n_x) = (0, 1, 0)$')
        plt.plot(sim_times_ar, be_ar2, label=r'$(n_y, n_z, n_x) = (0, 2, 0)$')
        plt.plot(sim_times_ar, be_ar3, label=r'$(n_y, n_z, n_x) = (0, 0, 0)$')
        if (True):
            ln_be_ar = np.log(be_ar1)         
            t_lin = sim_times_ar
            be_lin = ln_be_ar[:, 0]
            print(np.shape(t_lin))
            print(np.shape(be_lin))
            slope1, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_lin, be_lin)
            print('R2 val: ' + str(r_value**2))
            print('slope: ' + str(slope1))
            print('intercept: ' + str(intercept))      
            be_reg = np.exp(slope1 * np.array(t_lin) + intercept)
            plt.plot(t_lin, be_reg, color='k', linestyle='--', linewidth=2)

            fig.text(.2, .02, r'$(n_y, n_z, n_x) = (0, 1, 0)$: Growth rate = ' + str(round(slope1, 6)), ha='center')

            # ln_be_ar = np.log(be_ar2)
            # t_cutoff = 0
            # i_cutoff = -1
            # for i, t in enumerate(sim_times_ar):
            #     if (t > t_cutoff):
            #         i_cutoff = i
            #         break
            # t_cutoff2 = 130
            # i_cutoff2 = -1
            # for i, t in enumerate(sim_times_ar):
            #     if (t > t_cutoff2):
            #         i_cutoff2 = i
            #         break            
            # t_lin = sim_times_ar[i_cutoff:i_cutoff2]
            # be_lin = ln_be_ar[i_cutoff:i_cutoff2]
            # slope2, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_lin, be_lin)
            # print('R2 val: ' + str(r_value**2))
            # print('slope: ' + str(slope2))
            # print('intercept: ' + str(intercept))      
            # be_reg = np.exp(slope2 * np.array(t_lin) + intercept)
            # plt.plot(t_lin, be_reg, color='k', linestyle='--', linewidth=2)

            # fig.text(.2, -.05, r'$k_y, k_z \, = \, $' + str(ky2) + ',' + str(kz2) + ': Growth rate = ' + str(round(slope, 4)), ha='center')
            # plt.plot(sim_times_ar, be_ar1, label=r'$k_y, k_z \, = \, $' + str(ky1) + ',' + str(kz1) + '; growth rate = ' + str(round(slope1.real, 4)))
            # plt.plot(sim_times_ar, be_ar2, label=r'$k_y, k_z \, = \, $' + str(ky2) + ',' + str(kz2) + '; growth rate = ' + str(round(slope2.real, 4)))

            # plt.legend()        # plt.xlim(0, 400)


        plt.xlabel(r'$t$')
        plt.ylabel(r'$\langle |\mathbf{b}|^2 \rangle_{\mathcal{D}}$')
        plt.legend()
        plt.title(r'$\rm{Re}^{-1} \, = \, \eta \, = \nu \, = \, ' + diff_str + '; \; S/S_C = 0.36$')
        plt.savefig(path + '/be_modes_' + write_suffix + '.png')

