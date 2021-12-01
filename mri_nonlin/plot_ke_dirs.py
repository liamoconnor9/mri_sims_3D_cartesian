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
        ke_data = pickle.load(open(path + '/ke_dirs.pick', 'rb'))
        ke_ar1 = ke_data['ke_ar1']
        ke_ar2 = ke_data['ke_ar2']
        ke_ar3 = ke_data['ke_ar3']
        sim_times_ar = ke_data['sim_times_ar']

        for index in range(start, start+count):
            field['g'] = file['tasks']['vx'][index]
            ke_totx = np.min(de.operators.integrate(field * field, 'x', 'y', 'z').evaluate()['g'])
        
            if (just_perts):
                field['g'] = file['tasks']['vy'][index]
            else:
                field['g'] = file['tasks']['vy'][index] + S * x_grid
            ke_toty = np.min(de.operators.integrate(field * field, 'x', 'y', 'z').evaluate()['g'])
        
            field['g'] = file['tasks']['vz'][index]
            ke_totz = np.min(de.operators.integrate(field * field, 'x', 'y', 'z').evaluate()['g'])
        
            ke_ar1.append(ke_totx)
            ke_ar2.append(ke_toty)
            ke_ar3.append(ke_totz)

            sim_times_ar.append(file['scales/sim_time'][index])
        ke_data['ke_ar1'] = ke_ar1
        ke_data['ke_ar2'] = ke_ar2
        ke_data['ke_ar3'] = ke_ar3
        ke_data['sim_times_ar'] = sim_times_ar
        pickle.dump(ke_data, open(path + '/ke_dirs.pick', 'wb'))


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
    write_data = False
    plot = True
    regress = False
    global just_perts
    just_perts = False
    args = docopt(__doc__)
    write_suffix = args['--suffix']

    filename = Path('mri_options.cfg')
    config = ConfigParser()
    config.read(str(filename))
 
    B = config.getfloat('parameters','B')
    R      =  config.getfloat('parameters','R')
    q      =  config.getfloat('parameters','q')
    global S
    S      = -R*B*np.sqrt(q)
    f      =  R*B/np.sqrt(q)
 
    if (write_data):
        output_path = pathlib.Path('').absolute()
        ke_data = {'ke_ar1' : [], 'ke_ar2' : [], 'ke_ar3' : [], 'sim_times_ar' : []}
        pickle.dump(ke_data, open(path + '/ke_dirs.pick', 'wb'))
   
        Nx = 32
        Ny = 64
        Nz = 64
        for i in range(1, 11):
            N = int(2**i)
            if "N" + str(N) in write_suffix:
                Nx = N // 2
                Nz = N
                Ny = N
                print("resolution provided in run suffix: Ny = Nz = " + str(N))
                break

        Lx = np.pi
        ar = 8

        global fields 
        global x_basis 
        global y_basis
        global z_basis
        global x_grid
        
        x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
        y_basis = de.Fourier('y', Ny, interval=(0, Lx * ar))
        z_basis = de.Fourier('z', Nz, interval=(0, Lx * ar))
        domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64)
        x_grid = domain.grid(2)

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
        diff_str_dict = {'diff1en2' : '10^{-2}', 'diff1en3' : '10^{-3}'}
        diff_str = diff_str_dict[write_suffix[:8]]
        ke_data = pickle.load(open(path + '/ke_dirs.pick', 'rb'))
        ke_ar1 = np.array(ke_data['ke_ar1']) / (np.pi ** 3 * 8 * 8)
        ke_ar2 = np.array(ke_data['ke_ar2']) / (np.pi ** 3 * 8 * 8)
        ke_ar3 = np.array(ke_data['ke_ar3']) / (np.pi ** 3 * 8 * 8)
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
        plt.plot(sim_times_ar, ke_ar1, label = r"$\langle v_x^2 \rangle$")
        if (just_perts):
            plt.plot(sim_times_ar, ke_ar2, label = r"$\langle v_y^2 \rangle$")
        else:
            plt.plot(sim_times_ar, ke_ar2, label = r"$\langle (v_y + Sx)^2 \rangle$")
        plt.plot(sim_times_ar, ke_ar3, label = r"$\langle v_z^2 \rangle$")

        plt.legend()        # plt.xlim(0, 400)


        plt.xlabel(r'$t$')
        plt.ylabel(r'Kinetic Energy')
        plt.legend()
        plt.title(r'$\rm{Re}^{-1} \, = \, \eta \, = \nu \, = \, ' + diff_str + '; \; S/S_C = 1.02$')
        plt.savefig(path + '/ke_dirs_' + write_suffix + '_clean2.png')
