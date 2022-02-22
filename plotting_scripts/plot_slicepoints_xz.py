"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.ioff()
from dedalus.extras import plot_tools
import logging
logger = logging.getLogger(__name__)

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

def main(filename, start, count, output):
    midplane(filename, start, count, output)
    # integral(filename, start, count, output)


def midplane(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    normal_dir = 'y'
    tasks = ['vx_mid'+normal_dir, 'vy_mid'+normal_dir, 'vz_mid'+normal_dir, 'bx_mid'+normal_dir, 'by_mid'+normal_dir, 'bz_mid'+normal_dir]
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'mid_{:06}.png'.format(write)
    # Layout
    nrows, ncols = 6, 1
    image = plot_tools.Box(4, 4 / arz)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call plotting helper (dset axes: [t, x, y, z])
                dset = file['tasks'][task]
                image_axes = (2, 3)
                data_slices = (index, 0, slice(None), slice(None))
                # if (index % 5 != 0):
                #     continue
                plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=True)

                # x = file['scales/x/1.0'][()]
                # y = file['scales/y/1.0'][()]
                # yy, xx = np.meshgrid(y.flatten(), x.flatten())
                # data = dset[data_slices]
                # plot = plt.pcolormesh(xx, yy, data)
                # cbar = plt.colorbar(plot, cax=axes, orientation='horizontal',
                #     ticks=ticker.MaxNLocator(nbins=5))
                # cbar.outline.set_visible(False)
                # axes.xaxis.set_ticks_position('top')                # plt.pcolor(dset[data_slices])
                # axes.xaxis.set_label_position('top')
                # plt.xlabel("x")
                # plt.ylabel("y")
                # plt.title(task)
                
                # plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=True)
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.48, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            if (index % 1 == 0):
                fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


# def integral(filename, start, count, output):
#     """Save plot of specified tasks for given range of analysis writes."""

#     # Plot settings
#     tasks = ['b integral x4']
#     scale = 2.5
#     dpi = 100
#     title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
#     savename_func = lambda write: 'int_{:06}.png'.format(write)
#     # Layout
#     nrows, ncols = 1, 1
#     image = plot_tools.Box(2, 2)
#     pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
#     margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

#     # Create multifigure
#     mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
#     fig = mfig.figure
#     # Plot writes
#     with h5py.File(filename, mode='r') as file:
#         for index in range(start, start+count):
#             for n, task in enumerate(tasks):
#                 # Build subfigure axes
#                 i, j = divmod(n, ncols)
#                 axes = mfig.add_axes(i, j, [0, 0, 1, 1])
#                 # Call plotting helper (dset axes: [t, x, y, z])
#                 dset = file['tasks'][task]
#                 image_axes = (2, 1)
#                 data_slices = (index, slice(None), slice(None), 0)
#                 plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=True, cmap='Greys')
#             # Add time title
#             title = title_func(file['scales/sim_time'][index])
#             title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
#             fig.suptitle(title, x=0.48, y=title_height, ha='left')
#             # Save figure
#             savename = savename_func(file['scales/write_number'][index])
#             savepath = output.joinpath(savename)
#             fig.savefig(str(savepath), dpi=dpi)
#             fig.clear()
#     plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    suffix = args['--output'][9:]

    global ar, ary, arz
    ar = get_param_from_suffix(suffix, "AR", 8)    
    ary = get_param_from_suffix(suffix, "ARy", ar)    
    arz = get_param_from_suffix(suffix, "ARz", ar)    

    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

