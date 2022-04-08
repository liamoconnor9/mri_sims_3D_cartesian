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
from OptimizationContext import OptimizationContext
import ForwardKDV
import BackwardKDV
import matplotlib.pyplot as plt
path = os.path.dirname(os.path.abspath(__file__))

# keys are forward variables
# items are (backward variables, adjoint initial condition function: i.e. ux(T) = func(ux_t(T)))
lagrangian_dict = {
    'u' : 'u_t'
}

class OptParams:
    def __init__(self, T, num_cp, dt):
        self.T = T
        self.num_cp = num_cp
        self.dt = dt
        self.dT = T / num_cp
        self.dt_per_cp = int(self.dT // dt)


T = 0.3
num_cp = 1.0
dt = 1e-2
epsilon_safety = 1.0
epsilon_max = 0.25
opt_params = OptParams(T, num_cp, dt)
show_forward = False

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

timestepper = d3.RK443
forward_solver = forward_problem.build_solver(timestepper)
backward_solver = backward_problem.build_solver(timestepper)

write_suffix = 'kdv0'

# Objective functions (fields):

# Gt is maximized over t in (0, T)
Gt = dist.Field(name='Gt') #empty field for no objective

# HT is maximized at t = T
path = os.path.dirname(os.path.abspath(__file__))
U_data = np.loadtxt(path + '/kdv_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.Field(name='U', bases=xbasis)
U['g'] = U_data
HT = (u - U)**2

HTx = d3.Differentiate(HT, xcoord)
ux = d3.Differentiate(u, xcoord)

# Adjoint source term: derivative of Gt wrt u
backward_source = "0"


# Adjoint ic: -derivative of HT wrt u(T)
# backward_ic = {'u_t' : HTx / ux}

HTS = []
nannorm_count = 0
opt = OptimizationContext(domain, xcoord, forward_solver, backward_solver, timestepper, lagrangian_dict, opt_params, None, write_suffix)
opt.x = x
n = 20
# ic = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.21*Lx))**2) / (2*n)
rand = np.random.RandomState(seed=42)
ic = rand.rand(*x.shape)
mu = 4.1
sig = 0.5
guess = -np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
# guess = ic.copy()
opt.ic['u']['g'] = 0.0
# opt.ic['u']['c'][:N//2] = 0.0
# U['g'] = opt.ic['u']['g']
backward_ic = {'u_t' : U - u}
opt.backward_ic = backward_ic
opt.HT = HT
opt.U_data = U_data
opt.build_var_hotel()
# opt.show = True

indices = []
HT_norms = []
dirs = []
dir = 0

from datetime import datetime
startTime = datetime.now()
for i in range(101):
    opt.show = False
    if (show_forward and i % 20 == 0):
        opt.show = True
    # U.change_scales(1)
    # U['g'] = opt.ic['u']['g']
    # opt.U_data = U['g'].copy()
    old_grad = backward_solver.state[0]['g'].copy()
    opt.loop()
    new_grad = backward_solver.state[0]['g'].copy()
    if (np.isnan(opt.HT_norm)):
        logger.info("nan norm")
        nannorm_count += 1
        break
    if (opt.HT_norm <= 1e-10):
        break

    indices.append(i)
    HT_norms.append(opt.HT_norm)

    if (i > 0 and HT_norms[-1] > HT_norms[-2]):
        # dir += 1
        opt_params = OptParams(opt_params.T, opt_params.num_cp, opt_params.dt / 2.0)
        opt.opt_params = opt_params
        opt.build_var_hotel()
        logger.warning('Gradient descent failed. Decreasing timestep to dt = {}'.format(opt_params.dt))
        opt.loop_index -= 1
        old_grad = backward_solver.state[0]['g'].copy()
        opt.loop()
        new_grad = backward_solver.state[0]['g'].copy()
        HT_norms[-1] = opt.HT_norm    

    epsilon = epsilon_safety / (2**dir)

    gamma = 0.0001
    dirs.append(gamma)
    backward_solver.state[0].change_scales(1)
    
    if (i > 0):
        # https://en.wikipedia.org/wiki/Gradient_descent
        grad_diff = new_grad - old_grad
        x_diff = new_x - old_x
        gamma = epsilon * np.abs(np.dot(x_diff, grad_diff)) / np.dot(grad_diff, grad_diff)
        # gamma = min(epsilon_max, epsilon_safety * opt.HT_norm)

    new_x = opt.ic['u']['g'].copy() + gamma * backward_solver.state[0]['g']
    old_x = opt.ic['u']['g'].copy()
    opt.ic['u']['g'] = new_x

if not np.isnan(opt.HT_norm):
    HTS.append(HT_norms[-1])
logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
logger.info('Dir switches {}'.format(dir))
logger.info('####################################################')

plt.plot(indices, HT_norms, linewidth=2)
plt.yscale('log')
plt.ylabel('Error')
plt.xlabel('Loop Index')
# plt.show()
# plt.plot(indices, dirs)
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
plt.show()
plt.savefig(path + '/opt_ic.png')