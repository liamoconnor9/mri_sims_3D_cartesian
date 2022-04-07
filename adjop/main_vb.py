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
import ForwardVB
import BackwardVB
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

opt_params = OptParams(1.0, 1.0, 0.01, 1e-3)

# Bases
N = 256
Lx = 2.
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)
xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(-Lx / 2, Lx / 2), dealias=3/2)
domain = domain.Domain(dist, [xbasis])
dist = domain.dist


x = dist.local_grid(xbasis)

forward_problem = ForwardVB.build_problem(domain, xcoord, 1e-2)
backward_problem = BackwardVB.build_problem(domain, xcoord, 1e-2)

timestepper = d3.SBDF2
forward_solver = forward_problem.build_solver(timestepper)
backward_solver = backward_problem.build_solver(timestepper)

write_suffix = 'vb0'

# Objective functions (fields):

# Gt is maximized over t in (0, T)
Gt = dist.Field(name='Gt') #empty field for no objective

# HT is maximized at t = T
path = os.path.dirname(os.path.abspath(__file__))
U_data = np.loadtxt(path + '/vb_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.Field(name='U', bases=xbasis)
U['g'] = U_data
HT = np.abs(u - U)

# Adjoint source term: derivative of Gt wrt u
backward_source = "0"

# Adjoint ic: -derivative of HT wrt u(T)
backward_ic = {'u_t' : (U - u) / (((U - u)**2)**(0.5))}

HTS = []
nannorm_count = 0
for j in range(1):
    opt = OptimizationContext(domain, xcoord, forward_solver, backward_solver, timestepper, lagrangian_dict, opt_params, None, write_suffix)
    opt.x = x
    n = 20
    # ic = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.21*Lx))**2) / (2*n)
    rand = np.random.RandomState(seed=42)
    ic = rand.rand(*x.shape)
    guess = np.log(1 + np.cosh(n)**2/np.cosh(n*(x - 0.3))**2) / (2*n)
    # guess = ic.copy()
    opt.ic['u']['g'] = 0.0
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
    old_ic = opt.ic['u']['g'].copy()
    reduce_count = 0
    for i in range(101):
        opt.show = False
        if (True and i % 1 == 0):
            opt.show = True
        opt.loop()
        # if (i > 0):
        #     if (opt.HT_norm >= HT_norms[-1]):
        #         reduce_count += 1
        #         epsilon = 5e-3 / 2**reduce_count
        #         opt.ic['u']['g'] = old_ic + epsilon * old_grad
        #         continue
        indices.append(i)
        HT_norms.append(opt.HT_norm)
        if (np.isnan(opt.HT_norm)):
            logger.info("nan norm")
            nannorm_count += 1
            break
        dirs.append(dir)
        backward_solver.state[0].change_scales(1)
        if (i > 2 and i % 10 == 0 and HT_norms[-1] > min(HT_norms[-9:-2])):
            reduce_count += 1

        # epsilon = -opt.HT_norm / 100 / 2**dir
        epsilon = 0.5 * opt.HT_norm / 1.2**dir

        opt.ic['u']['g'] = opt.ic['u']['g'].copy() + epsilon * backward_solver.state[0]['g']
    if not np.isnan(opt.HT_norm):
        HTS.append(HT_norms[-1])
    logger.info('####################################################')
    logger.info('COMPLETED OPTIMIZATION RUN {}'.format(j))
    logger.info('Dir switches {}'.format(dir))
    logger.info('Reduce Count {}'.format(reduce_count))
    logger.info('####################################################')

    plt.plot(indices, HT_norms, linewidth=2)
    plt.yscale('log')
    plt.ylabel('Error')
    plt.xlabel('Loop Index')
    plt.show()
    # plt.plot(indices, dirs)
    # plt.show()
    n = 20
    soln = np.log(1 + np.cosh(n)**2/np.cosh(n*(x))**2) / (2*n)
    approx = opt.ic['u']['g'].flatten()
    grad = backward_solver.state[0]['g'].copy()
    grad *= np.max(np.abs(approx)) / np.max(np.abs(grad))
    plt.plot(x, approx, label="Optimized IC")
    plt.plot(x, soln, label="Real IC")
    plt.plot(x, grad, label="Gradient")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u(x, 0)$')
    plt.legend()
    plt.show()

# plt.hist(HTS)
# logger.info('HTS average = {}'.format(np.mean(HTS)))
# logger.info('nannorm count = {}'.format(nannorm_count))
# plt.show()