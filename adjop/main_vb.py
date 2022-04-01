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
HT = 0.5*(u - U)**2

# Adjoint source term: derivative of Gt wrt u
backward_source = "0"

# Adjoint ic: -derivative of HT wrt u(T)
backward_ic = {'u_t' : U - u}

HTS = []
nannorm_count = 0
for j in range(30):
    opt = OptimizationContext(domain, xcoord, forward_solver, backward_solver, timestepper, lagrangian_dict, opt_params, None, write_suffix)
    opt.x = x
    n = 20
    # ic = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.21*Lx))**2) / (2*n)
    ic = np.random.rand(*x.shape)
    opt.ic['u']['g'] = ic.copy()
    opt.ic['u']['c'][:N//2] = 0.0
    opt.backward_ic = backward_ic
    opt.HT = HT
    opt.U_data = U_data
    opt.build_var_hotel()
    # opt.show = True

    indices = []
    HT_norms = []
    dirs = []
    dir = 0
    for i in range(501):
        opt.show = False
        # if (i % 10 == 0):
            # opt.show = True
        opt.loop()
        indices.append(i)
        HT_norms.append(opt.HT_norm)
        if (np.isnan(opt.HT_norm)):
            logger.info("nan norm")
            nannorm_count += 1
            break
        dirs.append(dir)
        backward_solver.state[0].change_scales(1)
        if (i > 2 and HT_norms[-1] > HT_norms[-2]):
            dir += 1

        # epsilon = (-1)**dir / 10 / 10**dir
        if (i < 20):
            epsilon = (-1)**dir * opt.HT_norm / (100.0 - 5*i)
        elif (i < 100):
            epsilon = (-1)**dir * opt.HT_norm / 2**dir
        else:
            epsilon = 100 * (-1)**dir * opt.HT_norm / 2**dir
        #     epsilon = 4*np.log10(i) * (-1)**dir * opt.HT_norm

        opt.ic['u']['g'] = opt.ic['u']['g'].copy() + epsilon * backward_solver.state[0]['g']
    if not np.isnan(opt.HT_norm):
        HTS.append(HT_norms[-1])
    logger.info('####################################################')
    logger.info('COMPLETED OPTIMIZATION RUN {}'.format(j))
    logger.info('Dir switches {}'.format(dir))
    logger.info('####################################################')

    plt.plot(indices, HT_norms)
    plt.yscale('log')
    plt.show()
    plt.plot(indices, dirs)
    plt.show()
    sys.exit()

plt.hist(HTS)
logger.info('HTS average = {}'.format(np.mean(HTS)))
logger.info('nannorm count = {}'.format(nannorm_count))
plt.show()