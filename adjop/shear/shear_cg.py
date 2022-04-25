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
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser

from OptimizationContext import OptimizationContext
import ForwardShear
import BackwardShear
import matplotlib.pyplot as plt
from scipy import optimize

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

forward_solver = forward_problem.build_solver(d3.RK443)
backward_solver = backward_problem.build_solver(d3.SBDF4)

U = dist.VectorField(coords, name='U', bases=bases)
slices = dist.grid_layout.slices(domain, scales=1)

# Populate U with end state of known initial condition
end_state_path = path + '/' + write_suffix + '/checkpoint_target/checkpoint_target_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
    logger.info('looding end state {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))

# Late time objective: HT is minimized at t = T
w2 = d3.skew(u)
W2 = d3.skew(U)
# HT = 0.5*(d3.dot(w2 - W2, w2 - W2))
HT = (d3.dot(u - U, u - U))

opt = OptimizationContext(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)

n = 20
mu = 4.1
sig = 0.5
guess = U['g'].copy()
# opt.ic['u'].fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise

opt.U = U
coeff = 0.0
opt.ic['u']['g'][0] = coeff * (1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1)))
opt.ic['u']['g'][1] += coeff * (0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01))
opt.ic['u']['g'][1] += coeff * (0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01))

# opt.ic['u']['g'][0] = 0.4 + 1/2 * (np.tanh((z-0.45)/0.1) - np.tanh((z+0.5)/0.1))
# opt.ic['u']['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01)
# opt.ic['u']['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01)

opt.ic['s'] = dist.Field(name='s', bases=bases)
opt.ic['s']['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))

# Adjoint ic: -derivative of HT wrt u(T)
# print(-HT.sym_diff(u))
# print(u.sym_diff(u))
# print(-HT.sym_diff(u).evaluate()['g'].shape)

# backward_ic = {'u_t' : -HT.sym_diff(u)}
# backward_ic = {'u_t' : 0*u}
backward_ic = {'u_t' : -(u - U)}
opt.backward_ic = backward_ic
backward_ic['s_t'] = dist.Field(name='s_t', bases=bases)
backward_ic['s_t']['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
opt.HT = HT
opt.z = z
indices = []
HT_norms = []
dt_reduce_index = 0

from datetime import datetime
startTime = datetime.now()

def f(x, *args):
    global x_old
    global u
    x_old = x
    if (opt.loop_index == opt_iters):
        raise opt.MyException({"message": "Achieved the proper number of loop index"})

    logger.info('start evaluating f..')
    opt.ic['u'].change_scales(1)
    opt.ic['u']['g'] = x.reshape((2,) + domain.grid_shape(scales=1))[:, slices[0], slices[1]]
    # CW.barrier()
    snapshots = opt.forward_solver.evaluator.add_file_handler(opt.run_dir + '/' + opt.write_suffix + '/snapshots_forward/snapshots_forward_loop' + str(opt.loop_index), sim_dt=0.001, max_writes=10, mode='overwrite')
    u = opt.forward_solver.state[0]
    s = opt.forward_solver.state[1]
    p = opt.forward_solver.state[2]
    snapshots.add_task(s, name='tracer')
    snapshots.add_task(p, name='pressure')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
    snapshots.add_task(d3.dot(ex, u), name='ux')
    snapshots.add_task(d3.dot(ez, u), name='uz')

    opt.loop_forward()
    logger.info('done evaluating f. norm = {}'.format(opt.HT_norm))
    return opt.HT_norm

def gradf(x, *args):
    global x_old
    logger.info('start grad f..{}'.format(CW.rank))
    if not np.allclose(x, x_old):
        f(x)

    snapshots_backward = opt.backward_solver.evaluator.add_file_handler(opt.run_dir + '/' + opt.write_suffix + '/snapshots_backward/snapshots_backward_loop' + str(opt.loop_index), sim_dt=-0.001, max_writes=10, mode='overwrite')
    u_t = opt.backward_solver.state[0]
    s_t = opt.backward_solver.state[1]
    p_t = opt.backward_solver.state[2]
    snapshots_backward.add_task(s_t, name='tracer')
    snapshots_backward.add_task(p_t, name='pressure')
    snapshots_backward.add_task(-d3.div(d3.skew(u_t)), name='vorticity')
    snapshots_backward.add_task(d3.dot(ex, u_t), name='ux')
    snapshots_backward.add_task(d3.dot(ez, u_t), name='uz')

    opt.loop_backward()
    logger.info('done grad f..{}'.format(CW.rank))
    opt.new_grad.change_scales(1)
    opt.loop_index += 1

    opt.new_grad['g']
    return opt.new_grad.allgather_data().flatten().copy()
    # return -opt.new_grad.allgather_data().flatten().copy()

opt.ic['u'].change_scales(1)
opt.ic['u']['g']
x0 = opt.ic['u'].allgather_data().flatten().copy()  # Initial guess.

def check_status(x):
    logger.info('completed scipy py iteration')
    CW.barrier()

# res1 = optimize.fmin_cg(f, x0, fprime=gradf, callback=check_status, retall=True, args=(opt,))
try:
    res1 = optimize.fmin_cg(f, x0, fprime=gradf, callback=check_status)
except opt.MyException as e:
    details = e.args[0]
    print(details["message"])
# logger.info(res1.message)