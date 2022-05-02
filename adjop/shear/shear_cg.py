"""
Dedalus script for adjoint looping:
Given an end state (checkpoint_U), this script recovers the initial condition with no prior knowledge
Usage:
    shear_cg.py <config_file> <run_suffix>
"""

from distutils.command.bdist import show_formats
import os
path = os.path.dirname(os.path.abspath(__file__))
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import sys
sys.path.append(path + "/..")
import h5py
import gc
import dedalus.public as d3
from dedalus.core import domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
# logger.setLevel(logging.info)
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser

from OptimizationContext import OptimizationContext
from ShearOptimization import ShearOptimization
import ForwardShear
import BackwardShear
import matplotlib.pyplot as plt
from scipy import optimize
from natsort import natsorted

args = docopt(__doc__)
filename = Path(args['<config_file>'])
write_suffix = args['<run_suffix>']

config = ConfigParser()
config.read(str(filename))

logger.info('Running shear_flow.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
opt_iters = config.getint('parameters', 'opt_iters')
num_cp = config.getint('parameters', 'num_cp')
handler_loop_cadence = config.getint('parameters', 'handler_loop_cadence')
add_handlers = config.getboolean('parameters', 'add_handlers')
guide_coeff = config.getfloat('parameters', 'guide_coeff')

Lx = config.getfloat('parameters', 'Lx')
Lz = config.getfloat('parameters', 'Lz')
Nx = config.getint('parameters', 'Nx')
Nz = config.getint('parameters', 'Nz')

Reynolds = config.getfloat('parameters', 'Re')
T = config.getfloat('parameters', 'T')
dt = config.getfloat('parameters', 'dt')

dealias = 3/2
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
coords.name = coords.names

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
x, z = dist.local_grids(bases[0], bases[1])
ex, ez = coords.unit_vector_fields(dist)

domain = domain.Domain(dist, bases)
dist = domain.dist

forward_problem = ForwardShear.build_problem(domain, coords, Reynolds)
backward_problem = BackwardShear.build_problem(domain, coords, Reynolds)

# forward, and corresponding adjoint variables (fields)
u = forward_problem.variables[0]
u_t = backward_problem.variables[0]
lagrangian_dict = {u : u_t}

forward_solver = forward_problem.build_solver(d3.RK222)
backward_solver = backward_problem.build_solver(d3.RK222)

opt = ShearOptimization(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)
opt.opt_iters = opt_iters
opt.add_handlers = add_handlers
opt.handler_loop_cadence = handler_loop_cadence

U = dist.VectorField(coords, name='U', bases=bases)
S = dist.Field(name='S', bases=bases)
slices = dist.grid_layout.slices(domain, scales=1)
opt.slices = slices

# Populate U with end state of known initial condition
end_state_path = path + '/' + write_suffix + '/checkpoint_target/checkpoint_target_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
    S['g'] = f['tasks/s'][-1, :, :][slices[0], slices[1]]
    logger.info('looding end state {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))
# sys.exit()
opt.S = S
# opt.ic['u'].fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise

opt.ic['u']['g'][0] = guide_coeff * (1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1)))
opt.ic['u']['g'][1] += guide_coeff * 0.1 * np.sin(2*np.pi*x/Lx)
opt.ic['u']['g'][1] += guide_coeff * 0.1 * np.sin(2*np.pi*x/Lx)

#old ic
# opt.ic['u']['g'][0] = coeff * (1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1)))
# opt.ic['u']['g'][1] += coeff * (0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01))
# opt.ic['u']['g'][1] += coeff * (0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01))


# restart_dir = path + '/' + write_suffix + '/checkpoints'
# checkpoint_names = [name for name in os.listdir(restart_dir) if 'loop' in name]
# last_checkpoint = natsorted(checkpoint_names)[-1]
# print(last_checkpoint)
# sys.exit()
# last_checkpoint_path = restart_dir + '/' + last_checkpoint
# write, last_dt = solver.load_state(last_checkpoint_path, -1)
# logger.info("loading previous state")

# # Timestepping and output
# dt = last_dt
# fh_mode = 'append'


opt.ic['s'] = dist.Field(name='s', bases=bases)
opt.ic['s']['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))

# Late time objective: objectiveT is minimized at t = T
# w2 = d3.div(d3.skew(u))
dx = lambda A: d3.Differentiate(A, coords['x'])
dz = lambda A: d3.Differentiate(A, coords['z'])
ux = u @ ex
uz = u @ ez
w = dx(uz) - dz(ux)
Ux = U @ ex
Uz = U @ ez
W = dx(Uz) - dz(Ux)
# W2 = d3.div(d3.skew(U))

objectiveT = (w - W)**2
# objectiveT = d3.dot(u - U, u - U)
opt.set_objectiveT(objectiveT)
opt.backward_ic['u_t'] = -2.0*d3.skew(d3.grad((w - W)))

opt.metricsT['u_error'] = d3.dot(u - U, u - U)
opt.metricsT['omega_error'] = (w - W)**2

# temp = opt.backward_ic['u_t']
# logger.info(temp)
# sys.exit()

# opt.backward_ic['s_t'] = opt.ic['s'].copy()

opt.backward_ic['s_t'] = dist.Field(name='s_t', bases=bases)
opt.backward_ic['s_t']['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))

from datetime import datetime
startTime = datetime.now()

opt.ic['u'].change_scales(1)
opt.ic['u']['g']
x0 = opt.ic['u'].allgather_data().flatten().copy()  # Initial guess.

def check_status(x):
    logger.info('completed scipy py iteration')
    CW.barrier()

def newton_descent(fun, x0, args, **kwargs):
    gamma = 0.001
    maxiter = kwargs['maxiter']
    jac = kwargs['jac']
    for i in range(maxiter):
        f = fun(x0)
        gradf = jac(x0)
        x0 -= gamma * gradf
    logger.info('success')
    logger.info('maxiter = {}'.format(maxiter))
    return optimize.OptimizeResult(x=x0, success=True, message='beep boop')

# logging.basicConfig(filename='/path/to/your/log', level=....)
logging.basicConfig(filename = opt.run_dir + '/' + opt.write_suffix + '/log.txt')
try:
    options = {'maxiter' : opt_iters}
    res1 = optimize.minimize(opt.loop_forward, x0, jac=opt.loop_backward, options=options, callback=check_status, tol=1e-8, method='L-BFGS-B')
    # res1 = optimize.minimize(opt.loop_forward, x0, jac=opt.loop_backward, options=options, callback=check_status, tol=1e-8, method=newton_descent)
    logger.info(res1)
    sys.exit()
except opt.LoopIndexException as e:
    details = e.args[0]
    logger.info(details["message"])
except opt.NanNormException as e:
    details = e.args[0]
    logger.info(details["message"])
# except Exception as e:
#     logger.info('Unknown exception occured: {}'.format(e))
logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
logger.info('BEST LOOP INDEX {}'.format(opt.best_index))
logger.info('BEST objectiveT {}'.format(opt.best_objectiveT))
logger.info('####################################################')
