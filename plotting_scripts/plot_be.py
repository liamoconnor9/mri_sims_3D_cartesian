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
import publication_settings
import logging
logger = logging.getLogger(__name__)
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
plt.gcf().subplots_adjust(left=0.15)

def get_param_from_suffix(suffix, param_prefix, default_param):
    required = np.isnan(default_param)
    prefix_index = suffix.find(param_prefix)
    if (prefix_index == -1):
        if (required):
            logger.warning("Required parameter " + param_prefix + ": value not provided in write suffix " + suffix)
            raise 
        else:
            logger.info("Using default parameter: " + param_prefix + " = " + str(default_param))
            return default_param
    else:
        try:
            val_start_index = prefix_index + len(param_prefix)
            end_ind = suffix[val_start_index:].find("_")
            if (end_ind != -1):
                val_end_index = val_start_index + suffix[val_start_index:].find("_")
            else:
                val_end_index = val_start_index + len(suffix[val_start_index:])
            val_str = suffix[val_start_index:val_end_index]
            en_ind = val_str.find('en')
            if (en_ind != -1):
                magnitude = -int(val_str[en_ind + 2:])
                val_str = val_str[:en_ind]
            else:
                e_ind = val_str.find('e')
                if (e_ind != -1):
                    magnitude = int(val_str[e_ind + 1:])
                    val_str = val_str[:e_ind]
                else:
                    magnitude = 0.0

            p_ind = val_str.find('p')
            div_ind = val_str.find('div')
            if (p_ind != -1):
                whole_val = val_str[:p_ind]
                decimal_val = val_str[p_ind + 1:]
                param = float(whole_val + '.' + decimal_val) * 10**(magnitude)
            elif (div_ind != -1):
                num_str = val_str[:div_ind]
                pi_ind = num_str.find('PI')
                if (pi_ind != -1):
                    num = np.pi * int(num_str[:pi_ind])
                else:
                    num = int(num_str)
                den = int(val_str[div_ind + 3:])
                param = num / den 
            else:
                param = float(val_str) * 10**(magnitude)  
            logger.info("Parameter " + param_prefix + " = " + str(param) + " : provided in write suffix")
            return param
        except Exception as e: 
            if (required):
                logger.warning("Required parameter " + param_prefix + ": failed to parse from write suffix")
                logger.info(e)
                raise 
            else:
                logger.info("Suffix parsing failed! Using default parameter: " + param_prefix + " = " + str(default_param))
                logger.info(e)
                return default_param

def title_string(write_suffix):
    title_str = (r'$\nu \, = \,' + str(ν) + ', \, \eta \, = \, $' + str(η) + r'$; \; S/S_C = $' + str(SdivbySc))
    tau = get_param_from_suffix(write_suffix, 'tau', 0)
    if (tau != 0):
        title_str += r'$; \, \tau = $' + str(tau)
    if (get_param_from_suffix(write_suffix, 'noslip', 0)):
        title_str += '; No Slip BCs'
    else:
        title_str += '; Stress Free BCs'
    return title_str

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        be_data = pickle.load(open(path + '/be_data_' + write_suffix + '.pick', 'rb'))
        be_arx = be_data['be_arx']
        be_ary = be_data['be_ary']
        be_arz = be_data['be_arz']
        # be_max_ar = be_data['be_max_ar']
        sim_times_ar = be_data['sim_times_ar']
        for index in range(start, start+count):
            be_avgx = file['tasks']['be_x'][index][0, 0, 0]
            be_avgy = file['tasks']['be_y'][index][0, 0, 0]
            be_avgz = file['tasks']['be_z'][index][0, 0, 0]
            be_arx.append(be_avgx)
            be_ary.append(be_avgy)
            be_arz.append(be_avgz)
            sim_times_ar.append(file['scales/sim_time'][index])
        be_data['be_arx'] = be_arx
        be_data['be_ary'] = be_ary
        be_data['be_arz'] = be_arz
        be_data['sim_times_ar'] = sim_times_ar
        pickle.dump(be_data, open(path + '/be_data_' + write_suffix + '.pick', 'wb'))
        


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
    regress = False
    if (write_data):
        output_path = pathlib.Path('').absolute()
        be_data = {'be_arx' : [], 'be_ary' : [], 'be_arz' : [], 'sim_times_ar' : []}
        pickle.dump(be_data, open(path + '/be_data_' + write_suffix + '.pick', 'wb'))

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

        diffusivities = get_param_from_suffix(write_suffix, "viff", np.NaN)
        ν = get_param_from_suffix(write_suffix, 'nu', diffusivities)
        η = get_param_from_suffix(write_suffix, 'eta', diffusivities)
        R = get_param_from_suffix(write_suffix, "R", np.NaN)

        be_data = pickle.load(open(path + '/be_data_' + write_suffix + '.pick', 'rb'))
        be_arx = be_data['be_arx']
        be_ary = be_data['be_ary']
        be_arz = be_data['be_arz']
        sim_times_ar = be_data['sim_times_ar']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        shave_ind = 10
        plt.plot(sim_times_ar[:-shave_ind], np.array(be_arx[:-shave_ind]), color='darkblue', label=r'$\langle b^2_x \rangle_{\mathcal{D}}$')
        plt.plot(sim_times_ar[:-shave_ind], np.array(be_ary[:-shave_ind]), color='darkred', label=r'$\langle b^2_y \rangle_{\mathcal{D}}$')
        plt.plot(sim_times_ar[:-shave_ind], np.array(be_arz[:-shave_ind]), color='darkgreen', label=r'$\langle b^2_z \rangle_{\mathcal{D}}$')
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
            be_lin = ln_be_ar[i_cutoff:i_cutoff2]
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_lin, be_lin)
            print('R2 val: ' + str(r_value**2))
            print('slope: ' + str(slope))
            print('intercept: ' + str(intercept))      
            be_reg = np.exp(slope * np.array(t_lin) + intercept)
            plt.plot(t_lin, be_reg, color='k', linestyle='--', linewidth=2, label='Best fit')

            fig.text(.2, .05, 'Growth rate = ' + str(round(slope, 4)), ha='center')

        q = 0.75
        S      = -R*np.sqrt(q)
        f      =  R/np.sqrt(q)
        SdivbySc = S / -1.0 * f
        plt.xlabel(r'$t$')
        plt.ylabel('Average Energy')

        plt.title(title_string(write_suffix))
        plt.legend()        # plt.xlim(0, 400)
        plt.savefig(path + '/be_nonlin_' + write_suffix + '.png')
