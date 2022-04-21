from distutils.command.bdist import show_formats
import os
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(path + "/..")
from OptimizationContext import OptimizationContext
import h5py
import gc
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
import ForwardKDV
import BackwardKDV
import matplotlib.pyplot as plt

T = 5.0
num_cp = 1
dt = 5e-3

epsilon_safety = default_gamma = 0.6
gain = 1.0
show_forward = False
cadence = 1
opt_iters = 50

# Bases
N = 256
Lx = 10.
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)

# xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(0, Lx), dealias=3/2)
xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, Lx), dealias=3/2)

domain = Domain(dist, [xbasis])
dist = domain.dist

x = dist.local_grid(xbasis)
a = 0.01
b = 0.2
forward_problem = ForwardKDV.build_problem(domain, xcoord, a, b)
backward_problem = BackwardKDV.build_problem(domain, xcoord, a, b)

# Names of the forward, and corresponding adjoint variables
lagrangian_dict = {forward_problem.variables[0] : backward_problem.variables[0]}

forward_solver = forward_problem.build_solver(d3.RK222)
backward_solver = backward_problem.build_solver(d3.RK222)

write_suffix = 'kdv0'

path = os.path.dirname(os.path.abspath(__file__))
U_data = np.loadtxt(path + '/kdv_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.Field(name='U', bases=xbasis)
U['g'] = U_data

# HT is maximized at t = T (Late time objective)
HT = 0.5*(U - u)**2

HTS = []
nannorm_count = 0
opt = OptimizationContext(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.beta_calc = 'euler'
opt.set_time_domain(T, num_cp, dt)
opt.x = x
opt.use_euler = False
n = 20
mu = 4.1
sig = 0.5
guess = -np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

opt.ic['u']['g'] = guess

# Adjoint ic: -derivative of HT wrt u(T)
backward_ic = {'u_t' : -HT.sym_diff(u)}
opt.backward_ic = backward_ic
opt.HT = HT
opt.U_data = U_data
opt.build_var_hotel()

indices = []
HT_norms = []
dt_reduce_index = 0

from datetime import datetime
startTime = datetime.now()
# opt.show_cadence = 20
# for i in range(opt_iters):
#     if (i % 5 == 0):
#         opt.show = show_forward
#     else:
#         opt.show = False

#     opt.loop()

#     # if (opt.HT_norm <= 1e-10):
#     #     break

#     indices.append(i)
#     HT_norms.append(opt.HT_norm)
#     gamma = opt.compute_gamma(epsilon_safety)
#     # gamma = default_gamma
#     opt.descend(gamma)

# if not np.isnan(opt.HT_norm):
#     HTS.append(HT_norms[-1])
# logger.info('####################################################')
# logger.info('COMPLETED OPTIMIZATION RUN')
# logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
# logger.info('####################################################')

# # plt.plot(indices, HT_norms, linewidth=2)
# # plt.yscale('log')
# # plt.ylabel('Error')
# # plt.xlabel('Loop Index')
# # plt.savefig(path + '/error_kdv.png')
# # plt.close()

# # mu = 5.5
# # sig = 0.5
# # soln = 1*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
# # approx = opt.ic['u']['g'].flatten()

# # plt.plot(x, approx, label="Optimized IC")
# # plt.plot(x, soln, linestyle='--', label="Real IC")
# # plt.plot(x, guess, linestyle=':', label="Initial Guess")
# # plt.xlabel(r'$x$')
# # plt.ylabel(r'$u(x, 0)$')
# # plt.legend()
# # plt.savefig(path + '/opt_ic.png')
# # plt.show()

global x_old
def f(x):
    global x_old 
    x_old = x.copy()
    logger.info('start evaluating f..')
    opt.ic['u'].change_scales(1)
    opt.ic['u']['g'] = x.reshape(dist.grid_layout.local_shape(u.domain, scales=1)).copy()
    opt.loop_forward()
    logger.info('done evaluating f. norm = {}'.format(opt.HT_norm))
    return opt.HT_norm

def gradf(x, *args):
    global x_old
    logger.info('start grad f..')
    if not np.allclose(x, x_old):
        f(x)
    opt.evaluate_gradf()
    logger.info('done grad f..')
    opt.new_grad.change_scales(1)
    return -opt.new_grad['g'].flatten().copy()

x0 = opt.ic['u']['g'].flatten().copy()  # Initial guess.
# print(opt.ic['u']['g'].flatten().shape)
# logger.info(opt.ic['u']['g'][slices].flatten().shape)
# logger.info(opt.ic['u']['g'][slices].shape)
# sys.exit()
from scipy import optimize
res1 = optimize.minimize(f, x0, jac=gradf, method='L-BFGS-B')