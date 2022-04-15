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

T = 1.0
num_cp = 1
dt = 2.5e-3

Reynolds = 1e3
gamma = 5e-1
gamma_factor = 1.1
gain = 1
use_euler_gradient_descend = True
show_forward = False
cadence = 1
opt_iters = 1000

# Bases
Lx, Lz = 4, 1
Nx, Nz = 256, 128

dealias = 3/2
coords = d3.CartesianCoordinates('x', 'z')
coords.name = coords.names
dist = d3.Distributor(coords, dtype=np.float64)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.Chebyshev(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
x, z = dist.local_grids(bases[0], bases[1])

domain = domain.Domain(dist, bases)
dist = domain.dist

forward_problem = ForwardShear.build_problem(domain, coords, Reynolds)
backward_problem = BackwardShear.build_problem(domain, coords, Reynolds)

# Names of the forward, and corresponding adjoint variables
lagrangian_dict = {'u' : 'u_t'}

timestepper = d3.RK443
forward_solver = forward_problem.build_solver(timestepper)
backward_solver = backward_problem.build_solver(timestepper)

write_suffix = 'kdv0'

# HT is maximized at t = T
u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.VectorField(coords, name='U', bases=bases)
slices = dist.grid_layout.slices(domain, scales=1)
# print(slices)
# sys.exit()
with h5py.File(path + '/checkpoint_U/checkpoint_U_s1.h5') as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]

# Late time objective
HT = 0.5*(u - U)**2

HTS = []
nannorm_count = 0
opt = OptimizationContext(domain, coords, forward_solver, backward_solver, timestepper, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)

# This is our initial quess for the optimized initial condition
opt.ic['u']['g'] = 0

# Adjoint ic: -derivative of HT wrt u(T)
backward_ic = {'u_t' : gain*(U - u)}
opt.backward_ic = backward_ic
opt.HT = HT
opt.build_var_hotel()

opt.indices = []
opt.HT_norms = []

from datetime import datetime
startTime = datetime.now()
i = 0
addendum_str = ''

# opt.loop_index incremented in descend function
while opt.loop_index <= opt_iters:

    if (opt.loop_index % 10 == 8):
        linearity = opt.richardson_gamma(gamma)
        addendum_str = "linearity = {}; ".format(linearity)
    else:
        opt.loop()
        addendum_str = ''

    if (opt.HT_norm <= 1e-10):
        logger.info('Breaking optimization loop: error within tolerance. HT_norm = {}'.format(opt.HT_norm))
        break
    
    opt.descend(gamma, addendum_str=addendum_str)

    if (opt.loop_index > 1):
        performance = opt.step_performance
        if performance > 0.95:
            gamma *= gamma_factor
        elif performance < 0.8:
            gamma *= 1 / 2.0



logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
logger.info('####################################################')

plt.plot(opt.indices, opt.HT_norms, linewidth=2)
plt.yscale('log')
plt.ylabel('Error')
plt.xlabel('Loop Index')
plt.savefig(path + '/error_shear.png')
plt.close()