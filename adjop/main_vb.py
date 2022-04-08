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
    def __init__(self, T, num_cp, dt):
        self.T = T
        self.num_cp = num_cp
        self.dt = dt
        self.dT = T / num_cp
        self.dt_per_cp = int(self.dT // dt)

opt_params = OptParams(1.0, 1.0, 0.01)

# Bases
N = 256
Lx = 2.
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)
xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(-Lx / 2, Lx / 2), dealias=3/2)
domain = domain.Domain(dist, [xbasis])
dist = domain.dist
epsilon = 0.2

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

new_x = dist.Field(name='new_x', bases=xbasis)
new_grad = dist.Field(name='new_grad', bases=xbasis)
old_x = dist.Field(name='old_x', bases=xbasis)
old_grad = dist.Field(name='old_grad', bases=xbasis)

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
    for i in range(601):
        opt.show = False
        if (True and i % 100 == 0):
            opt.show = True
        old_grad['g'] = backward_solver.state[0]['g'].copy()
        opt.loop()
        new_grad['g'] = backward_solver.state[0]['g'].copy()
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
        if (i > 2 and HT_norms[-1] / HT_norms[-2] > 1.001):
            reduce_count += 1

            opt_params = OptParams(opt_params.T, opt_params.num_cp, opt_params.dt / 2.0)
            opt.opt_params = opt_params
            opt.build_var_hotel()
            logger.warning('Gradient descent failed. Decreasing timestep to dt = {}'.format(opt_params.dt))
            opt.loop_index -= 1
            old_grad['g'] = backward_solver.state[0]['g'].copy()
            opt.loop()
            new_grad['g'] = backward_solver.state[0]['g'].copy()
            HT_norms[-1] = opt.HT_norm    

        # epsilon = -opt.HT_norm / 100 / 2**dir
        gamma = 0.001
        if (i > 0):
            # https://en.wikipedia.org/wiki/Gradient_descent
            grad_diff = new_grad - old_grad
            x_diff = new_x - old_x
            gamma = epsilon * np.abs(d3.Integrate(x_diff * grad_diff).evaluate()['g'][0]) / (d3.Integrate(grad_diff * grad_diff).evaluate()['g'][0])
            # gamma = min(epsilon_max, epsilon_safety * opt.HT_norm)

        new_x.change_scales(1)
        old_x.change_scales(1)
        new_grad.change_scales(1)
        old_grad.change_scales(1)
        new_x['g'] = opt.ic['u']['g'].copy() + gamma * backward_solver.state[0]['g']
        old_x['g'] = opt.ic['u']['g'].copy()
        opt.ic['u']['g'] = new_x['g'].copy()
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