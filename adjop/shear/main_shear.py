from distutils.command.bdist import show_formats
import os
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import sys
import h5py
import gc
import dedalus.public as d3
from dedalus.core import domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
from OptimizationContext import OptimizationContext, OptParams
import ForwardShear
import BackwardShear
import matplotlib.pyplot as plt
path = os.path.dirname(os.path.abspath(__file__))

# keys are forward variables
# items are (backward variables, adjoint initial condition function: i.e. ux(T) = func(ux_t(T)))

T = 0.3
num_cp = 1
dt = 5e-3
opt_params = OptParams(T, num_cp, dt)

default_gamma = 3e-5
gain = 1e4
use_euler_gradient_descend = True
show_forward = False
cadence = 1
opt_iters = 501

# Bases
Lx, Lz = 1, 2
Nx, Nz = 128, 256

dealias = 3/2
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=np.float64)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
x, z = dist.local_grids(bases[0], bases[1])

domain = domain.Domain(dist, bases)
dist = domain.dist

Reynolds = 1e5
forward_problem = ForwardShear.build_problem(domain, coords, Reynolds)
backward_problem = BackwardShear.build_problem(domain, coords, Reynolds)

# Names of the forward, and corresponding adjoint variables
lagrangian_dict = {'u' : 'u_t'}

timestepper = d3.RK443
forward_solver = forward_problem.build_solver(timestepper)
backward_solver = backward_problem.build_solver(timestepper)

write_suffix = 'kdv0'

# HT is maximized at t = T
path = os.path.dirname(os.path.abspath(__file__))
U_data0 = np.loadtxt(path + '/shear_U0.txt')
U_data1 = np.loadtxt(path + '/shear_U1.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.VectorField(coords, name='U', bases=bases)

U['g'][0] = U_data0
U['g'][1] = U_data1

# Late time objective
HT = (u - U)**2

HTS = []
nannorm_count = 0
opt = OptimizationContext(domain, coords, forward_solver, backward_solver, timestepper, lagrangian_dict, opt_params, None, write_suffix)
# opt.x = x
opt.use_euler = use_euler_gradient_descend

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
backward_ic = {'u_t' : gain*(U - u)}
opt.backward_ic = backward_ic
opt.HT = HT
opt.build_var_hotel()

indices = []
HT_norms = []
dt_reduce_index = 0

from datetime import datetime
startTime = datetime.now()
for i in range(opt_iters):

    opt.loop()

    if (opt.HT_norm <= 1e-10):
        logger.info('Breaking optimization loop: error within tolerance. HT_norm = {}'.format(opt.HT_norm[0, 0]))
        break

    indices.append(i)
    HT_norms.append(opt.HT_norm[0, 0])
    opt.descend(default_gamma)

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