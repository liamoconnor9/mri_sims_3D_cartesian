"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<output> --dir=<dir>, --config=<config_file>, --suffix=<run_suffix>]

Options:
    --output=<output>  Output directory [default: ./frames]
    --dir=<dir> Dedalus run script directory
    --config=<config_file> simulation config file
    --suffix=<suff>  write suffix (run name)
"""

from docopt import docopt
from configparser import ConfigParser
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.ioff()
from dedalus.extras import plot_tools
import logging
import sys
logger = logging.getLogger(__name__)

def main(filename, start, count, output):
    midplane(filename, start, count, output)

def midplane(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    normal_dir = 'z'
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'mid_{:06}.png'.format(write)
    # Layout
    # if (round(ary) > 1):
    if (False):
        nrows, ncols = 6, 1
        tasks = ['vx_mid'+normal_dir, 'vy_mid'+normal_dir, 'vz_mid'+normal_dir, 'bx_mid'+normal_dir, 'by_mid'+normal_dir, 'bz_mid'+normal_dir]
    else:
        nrows, ncols = 3, 2
        tasks = ['vx_mid'+normal_dir, 'bx_mid'+normal_dir, 'vy_mid'+normal_dir, 'by_mid'+normal_dir, 'vz_mid'+normal_dir, 'bz_mid'+normal_dir]

    image = plot_tools.Box(2, 2 / ary)
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
                image_axes = (1, 3)
                data_slices = (index, slice(None), 0, slice(None))
                plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=True)

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

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    dir = args['--dir']
    suffix = args['--suffix']
    filename = dir + suffix + '/' + args['--config']
    config = ConfigParser()
    config.read(str(filename))

    global ar, ary, arz
    Ly = eval(config.get('parameters','Ly'))
    Lz = eval(config.get('parameters','Lz'))
    Lx = eval(config.get('parameters','Lx'))

    ary = Ly / Lx
    arz = Lz / Lx 

    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)