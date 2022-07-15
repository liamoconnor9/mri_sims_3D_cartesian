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
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import ForwardKDV
import BackwardKDV
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
import pickle
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
from KdvOptimization import KdvOptimization

filename = path + '/kdv_options.cfg'
config = ConfigParser()
config.read(str(filename))

logger.info('Running kdv_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
write_suffix = str(config.get('parameters', 'suffix'))
Lx = config.getfloat('parameters', 'Lx')
N = config.getint('parameters', 'Nx')

a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')
T = config.getfloat('parameters', 'T')
dt = config.getfloat('parameters', 'dt')
num_cp = config.getint('parameters', 'num_cp')

# Simulation Parameters
dealias = 3/2
dtype = np.float64

opt_iters = config.getint('parameters', 'opt_iters')
method = str(config.get('parameters', 'scipy_method'))
euler_safety = config.getfloat('parameters', 'euler_safety')
gamma_init = config.getfloat('parameters', 'gamma_init')
R = config.getfloat('parameters', 'R')
modes_dim = config.getint('parameters', 'modes_dim')

periodic = config.getboolean('parameters', 'periodic')
restart = config.getboolean('parameters', 'restart')
show_forward = config.getboolean('parameters', 'show')
show_iter_cadence = config.getint('parameters', 'show_iter_cadence')
show_loop_cadence = config.getint('parameters', 'show_loop_cadence')

sig = config.getfloat('parameters', 'sig_ic')
mu = config.getfloat('parameters', 'mu_ic')
ic_scale = config.getfloat('parameters', 'ic_scale')

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64, comm=MPI.COMM_SELF)

if periodic:
    xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, Lx), dealias=3/2)
else:
    xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(0, Lx), dealias=3/2)

domain = Domain(dist, [xbasis])
dist = domain.dist

x = dist.local_grid(xbasis)
forward_problem = ForwardKDV.build_problem(domain, xcoord, a, b)
backward_problem = BackwardKDV.build_problem(domain, xcoord, a, b)

# Names of the forward, and corresponding adjoint variables
lagrangian_dict = {forward_problem.variables[0] : backward_problem.variables[0]}

forward_solver = forward_problem.build_solver(d3.RK443)
backward_solver = backward_problem.build_solver(d3.RK443)

opt = KdvOptimization(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.beta_calc = 'euler'
opt.set_time_domain(T, num_cp, dt)
opt.opt_iters = opt_iters
opt.gamma_init = gamma_init

opt.x_grid = x
opt.show_iter_cadence = show_iter_cadence
opt.show_loop_cadence = show_loop_cadence
opt.show = show_forward

soln = 2*np.sin(2*np.pi*x / Lx) + 2*np.sin(4*np.pi*x / Lx)
# soln = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
soln_f = dist.Field(name='soln_f', bases=xbasis)
soln_f['g'] = soln.reshape((1, N))

guess = 0.0

tracker_name = path + '/' + write_suffix + '/tracker_rank' + str(CW.rank) + '.pick'
opt.cp_internal_names = 'cp' + str(CW.rank)

if restart:
    opt.descent_tracker = {}
    opt.descent_tracker['objectiveT'] = []
    opt.descent_tracker['x'] = []
    load_ind = 0
else:
    with open(tracker_name, 'rb') as file:
        opt.descent_tracker = pickle.load(file)
    opt.loop_index = len(opt.descent_tracker['x'])
    load_ind = opt.loop_index

U_data = np.loadtxt(path + '/kdv_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')
U = dist.Field(name='U', bases=xbasis)
U['g'] = U_data
opt.U_data = U_data

objectiveT = 0.5*(U - u)**2
opt.set_objectiveT(objectiveT)

mode1 = dist.Field(name='mode1', bases=xbasis)
mode1['g'] = np.sin(2*np.pi*x / Lx)

mode2 = dist.Field(name='mode2', bases=xbasis)
mode2['g'] = np.sin(4*np.pi*x / Lx)

opt.ic['u']['g'] = ic_scale*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

np.random.seed(CW.rank)
coeffs = 2*np.random.rand(modes_dim) - 1.0
coeffs *= R / np.sqrt(np.sum(coeffs**2))
for kx, coeff in enumerate(coeffs):
    opt.ic['u']['g'] += coeff*np.cos((kx + 1)*2*np.pi*x / Lx)

# opt.ic['u']['g'] = 2.987*mode1['g'] + 7.3*mode2['g']

# # opt.metrics0['A1'] = opt.ic['u']*mode1 / Lx * 2.0
# # opt.metrics0['A2'] = opt.ic['u']*mode2 / Lx * 2.0
# # opt.metrics0['Arem'] = opt.ic['u']*opt.ic['u'] - (opt.metrics0['A1']**2 + opt.metrics0['A2']**2) * Lx * 2.0

# opt.track_metrics()

def euler_descent(fun, x0, args, **kwargs):
    # gamma = 0.001
    # maxiter = kwargs['maxiter']
    maxiter = opt_iters
    jac = kwargs['jac']
    f = np.nan
    gamma = np.nan
    refinement_index = 0
    base = 2
    substeps_num = base**refinement_index
    substeps_left = 1
    for i in range(opt.loop_index, maxiter):
        old_f = f
        f, gradf = opt.loop(x0)
        old_gamma = gamma
        gamma = opt.compute_gamma(euler_safety)
        if (euler_safety == 0.0):
            gamma = gamma_init
        opt.metricsT_norms['gamma'] = gamma
        opt.metricsT_norms['ref_ind'] = refinement_index
        opt.metricsT_norms['substeps_left'] = substeps_left

        if (load_ind + 2 <= opt.loop_index):
            step_p = (old_f - f) / old_gamma / (opt.old_grad_sqrd**0.5) * substeps_num
            opt.metricsT_norms['step_p'] = step_p
            if (step_p < 0.9):
                refinement_index += 1
                substeps_num = base**refinement_index
                substeps_left = base**refinement_index

            
        if i % 1 == 0 and substeps_left == 1:
            opt.descent_tracker['objectiveT'].append(f)
            opt.descent_tracker['x'].append(x0.copy())
            with open(tracker_name, 'wb') as file:
                pickle.dump(opt.descent_tracker, file)
            logger.info('wrote to tracker')

        x0 -= 1e4 * gamma * gradf / (opt.new_grad_sqrd**0.5) / substeps_num
        if (substeps_num > 1):
            if (substeps_left > 1):
                substeps_left -= 1
            else:
                substeps_left = substeps_num
    logger.info('success')
    logger.info('maxiter = {}'.format(maxiter))
    return optimize.OptimizeResult(x=x0, success=True, message='beep boop')

if (method == "euler"):
    method = euler_descent

startTime = datetime.now()
try:
    tol = 1e-10
    options = {'maxiter' : opt_iters, 'ftol' : tol, 'gtol' : tol}
    if restart:
        x0 = opt.ic['u']['g'].flatten().copy()  # Initial guess.
    else:
        x0 = opt.descent_tracker['x'][-1].copy()
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

# forward_ts = type(opt.forward_solver.timestepper).__name__
# backward_ts = type(opt.backward_solver.timestepper).__name__
# plt.plot(x, opt.new_grad['g'], label='{}, {}'.format(forward_ts, backward_ts))
# plt.plot(x, delta, label='apriori gradient')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel(r'$\mu(x, 0)$')
# plt.title('Gradient')
# plt.savefig(path + '/grad_u.png')
# plt.show()

# print(x0)
x0 = opt.best_x
plt.plot(x, x0, label='Optimized IC')
plt.plot(x, soln, linestyle=':', label='Target IC')
# plt.plot(x, guess, linestyle='--', label='Initial Guess')
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$u(x, 0)$')
plt.savefig(path + '/ics.png')
plt.show()
# plt.close()

# plt.plot(opt.objectiveT_norms)
# plt.xlabel('loop index')
# plt.ylabel(r'$0.5(u(T) - U(T))^2$')
# plt.title('Error')
# plt.savefig(path + '/error.png')
# # plt.show()
