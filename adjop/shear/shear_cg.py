import numpy as np


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
from OptimizationContext import OptimizationContext
import ForwardShear
import BackwardShear
import matplotlib.pyplot as plt

# keys are forward variables
# items are (backward variables, adjoint initial condition function: i.e. ux(T) = func(ux_t(T)))

if len(sys.argv) > 1:
    write_suffix = sys.argv[1]
else:
    write_suffix = 'temp'
T = 1.0
num_cp = 1
dt = 0.001

opt_iters = 10

# Bases
Lx, Lz = 1, 2
Nx, Nz = 256, 512
Reynolds = 1e4

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

# Names of the forward, and corresponding adjoint variables
u = forward_problem.variables[0]
u_t = backward_problem.variables[0]
lagrangian_dict = {u : u_t}

forward_solver = forward_problem.build_solver(d3.RK443)
backward_solver = backward_problem.build_solver(d3.SBDF4)

U = dist.VectorField(coords, name='U', bases=bases)
slices = dist.grid_layout.slices(domain, scales=1)

# Populate U with end state of known initial condition
end_state_path = path + '/checkpoint_U/checkpoint_U_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
    logger.info('looding end state {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))

# Late time objective: HT is minimized at t = T
HT = 0.5*(d3.dot(u - U, u - U))

opt = OptimizationContext(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)

n = 20
mu = 4.1
sig = 0.5
guess = U['g'].copy()
# opt.ic['u'].fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise

opt.ic['u']['g'][0] = 0.49999 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
opt.ic['u']['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01)
opt.ic['u']['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01)

opt.ic['s'] = dist.Field(name='s', bases=bases)
opt.ic['s']['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))

# Adjoint ic: -derivative of HT wrt u(T)
# backward_ic = {'u_t' : -HT.sym_diff(u)}
backward_ic = {'u_t' : -(u - U)}
opt.backward_ic = backward_ic
backward_ic['s_t'] = dist.Field(name='s_t', bases=bases)
backward_ic['s_t']['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
opt.HT = HT

indices = []
HT_norms = []
dt_reduce_index = 0

from datetime import datetime
startTime = datetime.now()

global x_ol
u = opt.forward_solver.state[0]
def f(x):
    if opt.loop_index >= opt_iters:
        raise
    global x_old
    global u
    x_old = x
    logger.info('start evaluating f..')
    opt.ic['u'].change_scales(1)
    opt.ic['u']['g'] = x.reshape((2,) + dist.grid_layout.local_shape(u.domain, scales=1))

    snapshots = opt.forward_solver.evaluator.add_file_handler(opt.run_dir + '/' + opt.write_suffix + '/snapshots/snapshots_loop' + str(opt.loop_index), sim_dt=0.01, max_writes=10, mode='overwrite')
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
    logger.info('start grad f..')
    if not np.allclose(x, x_old):
        f(x)

    snapshots_backward = opt.backward_solver.evaluator.add_file_handler(opt.run_dir + '/' + opt.write_suffix + '/snapshots_backward/snapshots_backward_loop' + str(opt.loop_index), sim_dt=-0.01, max_writes=10, mode='overwrite')
    u_t = opt.backward_solver.state[0]
    s_t = opt.backward_solver.state[1]
    p_t = opt.backward_solver.state[2]
    snapshots_backward.add_task(s_t, name='tracer')
    snapshots_backward.add_task(p_t, name='pressure')
    snapshots_backward.add_task(-d3.div(d3.skew(u_t)), name='vorticity')
    snapshots_backward.add_task(d3.dot(ex, u_t), name='ux')
    snapshots_backward.add_task(d3.dot(ez, u_t), name='uz')

    opt.loop_backward()
    logger.info('done grad f..')
    opt.new_grad.change_scales(1)
    opt.loop_index += 1

    return -opt.new_grad['g'].flatten()

opt.ic['u'].change_scales(1)
x0 = opt.ic['u']['g'].flatten().copy()  # Initial guess.
# print(opt.ic['u']['g'].flatten().shape)
# logger.info(opt.ic['u']['g'][slices].flatten().shape)
# logger.info(opt.ic['u']['g'][slices].shape)
# sys.exit()
from scipy import optimize
res1 = optimize.minimize(f, x0, jac=gradf, method='CG', options={'disp': True})
logger.info(res1.message)