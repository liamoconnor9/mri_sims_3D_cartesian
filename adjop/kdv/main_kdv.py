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
import ForwardKDV
import BackwardKDV
import matplotlib.pyplot as plt
path = os.path.dirname(os.path.abspath(__file__))

# keys are forward variables
# items are (backward variables, adjoint initial condition function: i.e. ux(T) = func(ux_t(T)))

T = 3.0
num_cp = 1.0
dt = 1e-2
opt_params = OptParams(T, num_cp, dt)

epsilon_safety = 0.925
use_euler_gradient_descend = True
show_forward = False
cadence = 1
opt_iters = 21

# Bases
N = 256
Lx = 10.
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)

# xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(0, Lx), dealias=3/2)
xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, Lx), dealias=3/2)

domain = domain.Domain(dist, [xbasis])
dist = domain.dist

x = dist.local_grid(xbasis)
a = 0.01
b = 0.2
forward_problem = ForwardKDV.build_problem(domain, xcoord, a, b)
backward_problem = BackwardKDV.build_problem(domain, xcoord, a, b)

# Names of the forward, and corresponding adjoint variables
lagrangian_dict = {'u' : 'u_t'}

timestepper = d3.RK443
forward_solver = forward_problem.build_solver(timestepper)
backward_solver = backward_problem.build_solver(timestepper)

write_suffix = 'kdv0'

# HT is maximized at t = T
path = os.path.dirname(os.path.abspath(__file__))
U_data = np.loadtxt(path + '/kdv_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.Field(name='U', bases=xbasis)
U['g'] = U_data

# Late time objective
HT = (u - U)**2

HTS = []
nannorm_count = 0
opt = OptimizationContext(domain, xcoord, forward_solver, backward_solver, timestepper, lagrangian_dict, opt_params, None, write_suffix)
opt.x = x
opt.use_euler = use_euler_gradient_descend

n = 20
mu = 4.1
sig = 0.5
guess = -np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# guess = ic.copy()
opt.ic['u']['g'] = 0.0

# Adjoint ic: -derivative of HT wrt u(T)
backward_ic = {'u_t' : U - u}
opt.backward_ic = backward_ic
opt.HT = HT
opt.U_data = U_data
opt.build_var_hotel()
# opt.show = True

indices = []
HT_norms = []
dt_reduce_index = 0

from datetime import datetime
startTime = datetime.now()
for i in range(opt_iters):

    # opt.show = True
    # opt.show_backward = True
    
    # if (show_forward and i % cadence == 0):
    #     opt.show = True
    #     opt.show_backward = True

    # U.change_scales(1)
    # U['g'] = opt.ic['u']['g']
    # opt.U_data = U['g'].copy()
    opt.loop()

    if (opt.HT_norm <= 1e-10):
        break

    indices.append(i)
    HT_norms.append(opt.HT_norm)

    if (i == 200):
        opt.update_timestep(opt.opt_params.dt / 2.0)
        logger.warning('Gradient descent failed. Decreasing timestep to dt = {}'.format(opt_params.dt))
        opt.loop_index -= 1

        opt.loop()
        HT_norms[-1] = opt.HT_norm
        dt_reduce_index = i

    gamma = opt.compute_gamma(epsilon_safety)

    opt.descend(gamma)

if not np.isnan(opt.HT_norm):
    HTS.append(HT_norms[-1])
logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
logger.info('####################################################')

plt.plot(indices, HT_norms, linewidth=2)
plt.yscale('log')
plt.ylabel('Error')
plt.xlabel('Loop Index')
plt.savefig(path + '/error_kdv.png')
plt.close()

mu = 5.5
sig = 0.5
soln = 1*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
approx = opt.ic['u']['g'].flatten()
plt.plot(x, approx, label="Optimized IC")
plt.plot(x, soln, linestyle='--', label="Real IC")
plt.plot(x, guess, linestyle=':', label="Initial Guess")
plt.xlabel(r'$x$')
plt.ylabel(r'$u(x, 0)$')
plt.legend()
plt.savefig(path + '/opt_ic.png')
plt.show()