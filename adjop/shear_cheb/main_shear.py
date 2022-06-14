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
# dt = 2.5e-4
dt = 0.001

gamma = gamma_init = 1e-1
default_gamma = 0.025
gamma_safety = 0.1
gamma_factor = 1.0
show_forward = False
cadence = 1
opt_iters = 300

# Bases
Lx, Lz = 1, 2
Nx, Nz = 256, 512

dealias = 3/2
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
coords.name = coords.names

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
x, z = dist.local_grids(bases[0], bases[1])

domain = domain.Domain(dist, bases)
dist = domain.dist

Reynolds = 1e4
forward_problem = ForwardShear.build_problem(domain, coords, Reynolds)
backward_problem = BackwardShear.build_problem(domain, coords, Reynolds)

# Names of the forward, and corresponding adjoint variables
u = forward_problem.variables[0]
u_t = backward_problem.variables[0]
lagrangian_dict = {u : u_t}

forward_solver = forward_problem.build_solver(d3.RK443)
backward_solver = backward_problem.build_solver(d3.SBDF4)

u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.VectorField(coords, name='U', bases=bases)
slices = dist.grid_layout.slices(domain, scales=1)

end_state_path = path + '/' + write_suffix + '/checkpoint_target/checkpoint_target_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
    logger.info('looding end state {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))

# Late time objective: HT is minimized at t = T
HT = 0.5*(d3.dot(u - U, u - U))
# print(HT.evaluate()['g'].shape)
# sys.exit()

HTS = []
nannorm_count = 0
opt = OptimizationContext(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)
opt.use_euler = True

n = 20
mu = 4.1
sig = 0.5
guess = U['g'].copy()
# opt.ic['u'].fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
opt.ic['u']['g'] = 0.0
ex, ez = opt.coords.unit_vector_fields(opt.domain.dist)
opt.ic['s'] = d3.dot(ex, opt.ic['u'])

# Adjoint ic: -derivative of HT wrt u(T)
# backward_ic = {'u_t' : -HT.sym_diff(u)}
backward_ic = {'u_t' : -(u - U)}
opt.backward_ic = backward_ic
backward_ic['s_t'] = d3.dot(ex, opt.backward_ic['u_t'])
opt.HT = HT

indices = []
HT_norms = []
dt_reduce_index = 0
# opt.add_handler(snapshots)

from datetime import datetime
startTime = datetime.now()
u = opt.forward_solver.state[0]

for i in range(opt_iters):

    # if (opt.loop_index % 10 == 8):
    #     linearity = opt.richardson_gamma(gamma)
    #     addendum_str = "linearity = {}; ".format(linearity)
    # else:
    opt.flow = d3.GlobalFlowProperty(opt.forward_solver, cadence=10)
    ex, ez = opt.coords.unit_vector_fields(opt.domain.dist)
    opt.flow.add_property((np.dot(u, ez))**2, name='w2')
    opt.flow.add_property(np.dot(u, u), name='ke')

    snapshots = opt.forward_solver.evaluator.add_file_handler(opt.run_dir + '/' + opt.write_suffix + '/snapshots_forward/snapshots_loop' + str(opt.loop_index), sim_dt=0.01, max_writes=10, mode='overwrite')
    u = opt.forward_solver.state[0]
    s = opt.forward_solver.state[1]
    p = opt.forward_solver.state[2]
    snapshots.add_task(s, name='tracer')
    snapshots.add_task(p, name='pressure')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    snapshots_backward = opt.backward_solver.evaluator.add_file_handler(opt.run_dir + '/' + opt.write_suffix + '/snapshots_backward/snapshots_backward_loop' + str(opt.loop_index), sim_dt=-0.01, max_writes=10, mode='overwrite')
    u_t = opt.backward_solver.state[0]
    s_t = opt.backward_solver.state[1]
    p_t = opt.backward_solver.state[2]
    snapshots_backward.add_task(s_t, name='tracer')
    snapshots_backward.add_task(p_t, name='pressure')
    snapshots_backward.add_task(-d3.div(d3.skew(u_t)), name='vorticity')

    opt.loop()

    if (opt.HT_norm <= 1e-10):
        logger.info('Breaking optimization loop: error within tolerance. HT_norm = {}'.format(opt.HT_norm))
        break
    
    if (opt.loop_index == 0):
        opt.grad_norm = 1.0
        # gamma = 1.0
    # else:
        # gamma = opt.compute_gamma(gamma_safety)


    opt.descend(gamma_init)

    if (gamma <= 1e-10):
        logger.info('Breaking optimization loop: gamma is negligible - increase resolution? gamma = {}'.format(gamma))
        break

logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
logger.info('####################################################')

plt.plot(indices, HT_norms, linewidth=2)
plt.yscale('log')
plt.ylabel('Error')
plt.xlabel('Loop Index')
plt.savefig(path + '/error_shear.png')
plt.close()
