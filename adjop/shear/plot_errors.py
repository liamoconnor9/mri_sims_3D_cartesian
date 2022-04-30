import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
path = os.path.dirname(os.path.abspath(__file__))
import sys
from dedalus.extras import plot_tools
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import pickle 

if len(sys.argv) > 1:
    write_suffix = sys.argv[1]
else:
    write_suffix = 'temp'

if len(sys.argv) > 3:
    t_dir, sh_dir, fr_dir = sys.argv[2], sys.argv[3], sys.argv[4]
else:
    t_dir, sh_dir, fr_dir = 'snapshots_target', 'snapshots_forward', 'frames_error'

with open(path + '/' + write_suffix + '/errors_data.pickle', 'rb') as handle:
    global_errors = pickle.load(handle)
    
maxindex = max(global_errors.keys())
for loop_ind in range(maxindex + 1):
    if not loop_ind in global_errors.keys():
        continue
    errors_data = global_errors[loop_ind]
    errors_unzipped = [list(t) for t in zip(*errors_data)]
    time = errors_unzipped[0]
    error = errors_unzipped[1]
    plt.plot(time, error, label='loop index {}'.format(loop_ind))
plt.yscale('log')
plt.xlabel('time')
plt.ylabel(r'$<|u(t) - U(t)|^2>$')
plt.title(write_suffix)
plt.legend()
plt.savefig(path + '/' + write_suffix + '/errors_test2.png')
