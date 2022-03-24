"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--suffix=write_string_suffix] 

"""

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
import dedalus.public as de
import publication_settings

matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
plt.gcf().subplots_adjust(left=0.15)

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        be_data = pickle.load(open(path + '/be_dirs.pick', 'rb'))
        be_ar1 = be_data['be_ar1']
        be_ar2 = be_data['be_ar2']
        be_ar3 = be_data['be_ar3']
        sim_times_ar = be_data['sim_times_ar']

        for index in range(start, start+count):
            field['g'] = file['tasks']['bx'][index]
            be_totx = np.min(de.operators.integrate(field * field, 'x', 'y', 'z').evaluate()['g'])
        
            field['g'] = file['tasks']['by'][index]
            be_toty = np.min(de.operators.integrate(field * field, 'x', 'y', 'z').evaluate()['g'])

            if (just_perts):
                field['g'] = file['tasks']['bz'][index]
            else:
                field['g'] = file['tasks']['bz'][index] + 1
            be_totz = np.min(de.operators.integrate(field * field, 'x', 'y', 'z').evaluate()['g'])

            be_ar1.append(be_totx)
            be_ar2.append(be_toty)
            be_ar3.append(be_totz)

            sim_times_ar.append(file['scales/sim_time'][index])
        be_data['be_ar1'] = be_ar1
        be_data['be_ar2'] = be_ar2
        be_data['be_ar3'] = be_ar3
            
        be_data['sim_times_ar'] = sim_times_ar
        pickle.dump(be_data, open(path + '/be_dirs.pick', 'wb'))
        


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
    regress = False
    global just_perts
    just_perts = False
    args = docopt(__doc__)
    write_suffix = args['--suffix']
    if (write_data):
        output_path = pathlib.Path('').absolute()
        be_data = {'be_ar1' : [], 'be_ar3' : [], 'be_ar2' : [], 'sim_times_ar' : []}
        pickle.dump(be_data, open(path + '/be_dirs.pick', 'wb'))

        Nx = 32
        Ny = 64
        Nz = 64
        Lx = np.pi
        ar = 8

        global fields 
        global x_basis 
        global y_basis
        global z_basis
        
        x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
        y_basis = de.Fourier('y', Ny, interval=(0, Lx * ar))
        z_basis = de.Fourier('z', Nz, interval=(0, Lx * ar))
        domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64)
        
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
        diff_str_dict = {'diff1en2' : '$10^{-2}$', 'viff1en2' : '$10^{-2}$', 'viff2en3' : r'$2 \times 10^{-3}$', 'viff1en3' : '$10^{-3}$', 'diff2en3' : r'$2 \times 10^{-3}$', 'diff3en3' : r'$3 \times 10^{-3}$', 'diff1en3' : '10^{-3}'}
        diff_str = diff_str_dict[write_suffix[:8]]
        be_data = pickle.load(open(path + '/be_dirs.pick', 'rb'))
        be_ar1 = np.array(be_data['be_ar1']) / (np.pi ** 3 * 8 * 8)
        be_ar2 = np.array(be_data['be_ar2']) / (np.pi ** 3 * 8 * 8)
        be_ar3 = np.array(be_data['be_ar3']) / (np.pi ** 3 * 8 * 8)
        sim_times_ar = be_data['sim_times_ar']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
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

        plt.plot(sim_times_ar, be_ar1, label = r"$\langle b^2_x \rangle$")
        plt.plot(sim_times_ar, be_ar2, label = r"$\langle b^2_y \rangle$")
        if (just_perts):
            plt.plot(sim_times_ar, be_ar3, label = r"$\langle b^2_z \rangle$")
        else:
            plt.plot(sim_times_ar, be_ar3, label = r"$\langle (b_z + B_0)^2 \rangle$")
        plt.legend()        # plt.xlim(0, 400)
        plt.xlabel(r'$t$')
        plt.ylabel(r'Magnetic Energy')
        plt.title(r'$\rm{Re}^{-1} \, = \, \eta \, = \nu \, = \, ' + diff_str + '; \; S/S_C = 1.02$')
        plt.savefig(path + '/be_dirs_' + write_suffix + '_clean.png')
