from distutils.command.bdist import show_formats
import os
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import sys
sys.path.append("..")
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
path = os.path.dirname(os.path.abspath(__file__))

# keys are forward variables
# items are (backward variables, adjoint initial condition function: i.e. ux(T) = func(ux_t(T)))

T = 20
num_cp = 1
dt = 1.25e-3

gamma = gamma_init = 5e-3
gamma_factor = 1.0
show_forward = False
cadence = 1
opt_iters = 200

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

Reynolds = 5e4
forward_problem = ForwardShear.build_problem(domain, coords, Reynolds)
backward_problem = BackwardShear.build_problem(domain, coords, Reynolds)

# Names of the forward, and corresponding adjoint variables
lagrangian_dict = {'u' : 'u_t'}

forward_solver = forward_problem.build_solver(d3.RK222)
backward_solver = backward_problem.build_solver(d3.SBDF2)

write_suffix = 'kdv0'

u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.VectorField(coords, name='U', bases=bases)
slices = dist.grid_layout.slices(domain, scales=1)

end_state_path = path + '/checkpoint_U/checkpoint_U_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
    logger.info('looding end state {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))

# Late time objective: HT is maximized at t = T
HT = (u - U)**2

HTS = []
nannorm_count = 0
opt = OptimizationContext(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)

n = 20
mu = 4.1
sig = 0.5
guess = -np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# guess = ic.copy()
# opt.ic['u']['g'][1] = 0.0

z_center = 0.49
# opt.ic['u']['g'][0] = 1/2 + 1/2 * (np.tanh((z-z_center)/0.1) - np.tanh((z+z_center)/0.1))
# opt.ic['u']['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-z_center)**2/0.01)
# opt.ic['u']['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+z_center)**2/0.01)

# Adjoint ic: -derivative of HT wrt u(T)
backward_ic = {'u_t' : (U - u)}
opt.backward_ic = backward_ic
opt.HT = HT
opt.build_var_hotel()

indices = []
HT_norms = []
dt_reduce_index = 0

from datetime import datetime
startTime = datetime.now()
for i in range(opt_iters):

    if (opt.loop_index % 10 == 8):
        linearity = opt.richardson_gamma(gamma)
        addendum_str = "linearity = {}; ".format(linearity)
    else:
        opt.loop()
        addendum_str = ''

    if (opt.HT_norm <= 1e-10):
        logger.info('Breaking optimization loop: error within tolerance. HT_norm = {}'.format(opt.HT_norm))
        break
    
    if (opt.loop_index == 0):
        opt.grad_norm = 1.0
        
    gamma = gamma_init / opt.grad_norm

    # if (opt.loop_index > 1):
    #     performance = opt.step_performance
    #     if performance > 0.95:
    #         gamma *= gamma_factor
    #     elif performance < 0.8:
    #         # gamma *= 1 / 10.0

    opt.descend(gamma, addendum_str=addendum_str)

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
