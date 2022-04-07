from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import os
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

# keys are forward variables
# items are (backward variables, adjoint initial condition function: i.e. ux(T) = func(ux_t(T)))
lagrangian_dict = {
    'u' : 'u_t'
}

class OptParams:
    def __init__(self, T, num_cp, dt, epsilon):
        self.T = T
        self.num_cp = num_cp
        self.dt = dt
        self.epsilon = epsilon
        self.dT = T / num_cp
        self.dt_per_cp = int(self.dT // dt)


T = 5
num_cp = 1.0
dt = 1e-2
epsilon = 0.0
opt_params = OptParams(T, num_cp, dt, epsilon)

# Bases
N = 256
Lx = 10.
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)
xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, Lx), dealias=3/2)
domain = domain.Domain(dist, [xbasis])
dist = domain.dist


x = dist.local_grid(xbasis)
a = 1e-6
b = 2e-1
forward_problem = ForwardKDV.build_problem(domain, xcoord, a, b)
backward_problem = BackwardKDV.build_problem(domain, xcoord, a, b)

timestepper = d3.SBDF2
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

# Adjoint source term: derivative of Gt wrt u
backward_source = "0"

# Adjoint ic: -derivative of HT wrt u(T)
backward_ic = {'u_t' : U - u}

HTS = []
nannorm_count = 0
opt = OptimizationContext(domain, xcoord, forward_solver, backward_solver, timestepper, lagrangian_dict, opt_params, None, write_suffix)
opt.x = x
n = 20
# ic = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.21*Lx))**2) / (2*n)
rand = np.random.RandomState(seed=42)
ic = rand.rand(*x.shape)
mu = 4.5
sig = 3.5
guess = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
# guess = ic.copy()
opt.ic['u']['g'] = 0
# opt.ic['u']['c'][:N//2] = 0.0
opt.backward_ic = backward_ic
opt.HT = HT
opt.U_data = U_data
opt.build_var_hotel()
# opt.show = True

indices = []
HT_norms = []
dirs = []
dir = 0
for i in range(100):
    opt.show = False
    if (False and i % 1 == 0):
        opt.show = True
    
    old_grad = backward_solver.state[0]['g'].copy()
    opt.loop()
    new_grad = backward_solver.state[0]['g'].copy()

    indices.append(i)
    HT_norms.append(opt.HT_norm)
    if (np.isnan(opt.HT_norm)):
        logger.info("nan norm")
        nannorm_count += 1
        break
    gamma = 1.0
    dirs.append(gamma)
    backward_solver.state[0].change_scales(1)
    if (i > 0):
        # https://en.wikipedia.org/wiki/Gradient_descent
        grad_diff = new_grad - old_grad
        x_diff = new_x - old_x
        gamma = np.abs(np.dot(x_diff, grad_diff)) / np.dot(grad_diff, grad_diff) / 5

    new_x = opt.ic['u']['g'].copy() + gamma * backward_solver.state[0]['g']
    old_x = opt.ic['u']['g'].copy()
    opt.ic['u']['g'] = new_x

if not np.isnan(opt.HT_norm):
    HTS.append(HT_norms[-1])
logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('Dir switches {}'.format(dir))
logger.info('####################################################')

plt.plot(indices, HT_norms, linewidth=2)
plt.yscale('log')
plt.ylabel('Error')
plt.xlabel('Loop Index')
plt.show()
plt.plot(indices, dirs)
plt.show()

mu = 5.5
sig = 1.5
soln = 0.1*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
approx = opt.ic['u']['g'].flatten()
plt.plot(x, approx, label="Optimized IC")
plt.plot(x, soln, label="Real IC")
# plt.plot(x, guess, label="Initial Guess")
plt.xlabel(r'$x$')
plt.ylabel(r'$u(x, 0)$')
plt.legend()
plt.show()