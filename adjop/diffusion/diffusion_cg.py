from distutils.command.bdist import show_formats
import os
path = os.path.dirname(os.path.abspath(__file__))
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import pickle
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
import ForwardDiffusion
import BackwardDiffusion
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
from DiffusionOptimization import DiffusionOptimization

filename = path + '/diffusion_options.cfg'
config = ConfigParser()
config.read(str(filename))

logger.info('Running diffusion_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
Lx = config.getfloat('parameters', 'Lx')
N = config.getint('parameters', 'Nx')

a = config.getfloat('parameters', 'a')
T = config.getfloat('parameters', 'T')
dt = config.getfloat('parameters', 'dt')
num_cp = config.getint('parameters', 'num_cp')

# Simulation Parameters
dealias = 3/2
dtype = np.float64

opt_iters = config.getint('parameters', 'opt_iters')
method = str(config.get('parameters', 'scipy_method'))

periodic = config.getboolean('parameters', 'periodic')
show_forward = config.getboolean('parameters', 'show')
epsilon_safety = default_gamma = 0.6
show_iter_cadence = config.getint('parameters', 'show_iter_cadence')
show_loop_cadence = config.getint('parameters', 'show_loop_cadence')

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)

if periodic:
    xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, Lx), dealias=3/2)
else:
    xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(0, Lx), dealias=3/2)

domain = Domain(dist, [xbasis])
dist = domain.dist

x = dist.local_grid(xbasis)
forward_problem = ForwardDiffusion.build_problem(domain, xcoord, a)
backward_problem = BackwardDiffusion.build_problem(domain, xcoord, a)

# Names of the forward, and corresponding adjoint variables
lagrangian_dict = {forward_problem.variables[0] : backward_problem.variables[0]}

forward_solver = forward_problem.build_solver(d3.RK443)
backward_solver = backward_problem.build_solver(d3.CNAB2)

write_suffix = 'diffusion0'

opt = DiffusionOptimization(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.beta_calc = 'euler'
opt.set_time_domain(T, num_cp, dt)
opt.opt_iters = opt_iters


opt.x_grid = x
opt.show_iter_cadence = show_iter_cadence
opt.show_loop_cadence = show_loop_cadence
opt.show = show_forward

n = 20
mu = 5.5
sig = 0.5
soln = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
soln_f = dist.Field(name='soln_f', bases=xbasis)
soln_f['g'] = soln.reshape((1, N))

delta = 0*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
guess = soln + delta


opt.ic['u']['g'] = guess.copy()
opt.w1 = dist.Field(name='w1', bases=xbasis)
# opt.w1['g'] = guess.copy()
opt.w1['g'] = np.cos(2*np.pi*(x/Lx - 0.5))
opt.w2 = dist.Field(name='w2', bases=xbasis)
# opt.w2['g'] = 1 - guess.copy()
opt.w2['g'] = np.cos(2*np.pi*4*(x/Lx - 0.5))
opt.x1 = []
opt.x2 = []

path = os.path.dirname(os.path.abspath(__file__))
U_data = np.loadtxt(path + '/diffusion_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.Field(name='U', bases=xbasis)
U['g'] = U_data
U['g'] = 0

objectiveT = 0.5*(U - u)**2
opt.set_objectiveT(objectiveT)

# opt.backward_ic = backward_ic
opt.U_data = U_data
opt.build_var_hotel()
opt.gamma_init = 1.0

def euler_descent(fun, x0, args, **kwargs):
    opt.gamma_init = 1.0
    maxiter = kwargs['maxiter']
    f = 0.0
    gamma = 0.1
    jac = kwargs['jac']
    for i in range(maxiter):
        old_f = f
        f = fun(x0)
        gradf = jac(x0)
        # old_gamma = gamma
        # gamma = opt.compute_gamma(1.0)
        # if i > 0:
            # step_p = (old_f - f) / old_gamma / opt.old_grad_sqrd
            # logger.info("step_p = {}".format(step_p))
            # opt.metricsT_norms['step_p'] = step_p

        x0 -= gamma * gradf
    return optimize.OptimizeResult(x=x0, success=True, message='beep boop')

if (method == "euler"):
    method = euler_descent

startTime = datetime.now()
try:
    tol = 1e-10
    options = {'maxiter' : opt_iters, 'ftol' : tol, 'gtol' : tol}
    x0 = opt.ic['u']['g'].flatten().copy()  # Initial guess.
    res1 = optimize.minimize(opt.loop_forward, x0, jac=opt.loop_backward, method=method, tol=tol, options=options)
    # res1 = optimize.minimize(opt.loop_forward, x0, jac=opt.loop_backward, method='L-BFGS-B', tol=tol, options=options)
    # res1 = optimize.minimize(opt.loop_forwaoh rd, x0, jac=opt.loop_backward, method=euler_descent, options=options)
    logger.info('scipy message {}'.format(res1.message))

except opt.LoopIndexException as e:
    details = e.args[0]
    logger.info(details["message"])
except opt.NanNormException as e:
    details = e.args[0]
    logger.info(details["message"])
logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
logger.info('BEST LOOP INDEX {}'.format(opt.best_index))
logger.info('BEST objectiveT {}'.format(opt.best_objectiveT))
logger.info('####################################################')

# diff = dist.Field(name='diff', bases=xbasis)
# diff['g'] = soln_f['g'] - opt.ic['u']['g']
# L1_integ = d3.Integrate(((diff)**2)**0.5).evaluate()
# logger.info('L1 error = {}'.format(L1_integ['g'].flat[0]))

forward_ts = type(opt.forward_solver.timestepper).__name__
backward_ts = type(opt.backward_solver.timestepper).__name__
# plt.plot(x, 0*opt.new_grad['g'], label='correct IC')
# plt.plot(x, opt.ic['u']['g'][0, :], label='optimized IC')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel(r'$u(x, 0)$')
# plt.title('ICs')
# plt.savefig(path + '/ics.png')
# plt.show()

with open(path + '/projections.pick', 'wb') as f:
    pickle.dump([opt.x1, opt.x2], f)

plt.scatter(opt.x1, opt.x2)
# plt.xscale('log')
# plt.yscale('log')
plt.show()
print('done')
# print(x0)
# x0 = opt.best_x
# plt.plot(x, x0, label='Optimized IC')
# plt.plot(x, soln, linestyle=':', label='Target IC')
# plt.plot(x, guess, linestyle='--', label='Initial Guess')
# plt.legend()
# plt.xlabel(r'$x$')
# plt.ylabel(r'$u(x, 0)$')
# plt.savefig(path + '/ics.png')
# plt.show()
# plt.close()

# plt.plot(opt.objectiveT_norms)
# plt.xlabel('loop index')
# plt.ylabel(r'$0.5(u(T) - U(T))^2$')
# plt.title('Error')
# plt.savefig(path + '/error.png')
# # plt.show()
