import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
path = os.path.dirname(os.path.abspath(__file__))
import sys
from dedalus.extras import plot_tools


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    loop_substr_index = filename.find('loop')
    # print(loop_substr_index)
    for i in range(4, 0, -1):
        # print(filename[loop_substr_index + 4:loop_substr_index + 4 + i])
        if filename[loop_substr_index + 4:loop_substr_index + 4 + i].isdigit():
            loop_ind = int(filename[loop_substr_index + 4:loop_substr_index + 4 + i])
            break

    # Plot settings
    tasks = ['tracer', 'pressure', 'vorticity', 'ux', 'uz']
    scale = 5
    dpi = 200
    title_func = lambda loop_index, sim_time: 'loop index = {}; t = {:.3f}'.format(loop_index, sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Layout
    nrows, ncols = 1, len(tasks)
    image = plot_tools.Box(1, 2)
    pad = plot_tools.Frame(0.2, 0, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)

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
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task, even_scale=True, visible_axes=False)
            # Add time title
            title = title_func(loop_ind, file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.45, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync
    import glob, os

    if len(sys.argv) > 1:
        write_suffix = sys.argv[1]
    else:
        write_suffix = 'temp'

    if len(sys.argv) > 3:
        sh_dir, fr_dir = sys.argv[2], sys.argv[3]
    else:
        sh_dir, fr_dir = 'snapshots', 'frames'

    for dir in os.listdir(path + '/' + write_suffix + '/' + sh_dir):
        full_dir = path + '/' + write_suffix + '/' + sh_dir + '/' + dir
        files_lst = []
        
        for file in glob.glob(full_dir + "/*.h5"):
            files_lst.append(str(file))
        output_dir = full_dir.replace(sh_dir, fr_dir)
        
        output_path = pathlib.Path(output_dir).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit_writes(files_lst, main, output=output_path)