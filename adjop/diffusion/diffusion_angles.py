from numpy.linalg import norm
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
import ForwardDiffusion
import BackwardDiffusion
import matplotlib.pyplot as plt
import matplotlib
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
from DiffusionOptimization import DiffusionOptimization

# sys.path.append("/home/liamo")
# import publication_settings
# matplotlib.rcParams.update(publication_settings.params)
# plt.rcParams.update({'figure.autolayout': True})

filename = path + '/diffusion_options.cfg'
config = ConfigParser()
config.read(str(filename))

logger.info('Running diffusion_ts.py with the following parameters:')
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
grads = []

eps = 1e-3
# timesteppers = [(d3.RK443, d3.SBDF2), (d3.RK443, d3.SBDF4)]
timesteppers = [(d3.RK443, d3.SBDF2)]
wavenumbers = list(range(1, 40))
angles = []
# wavenumbers = [1]
# timesteppers += [(d3.RK443, d3.RK222), (d3.RK443, d3.RK443)]
# timesteppers = [(d3.RK443, d3.SBDF1), (d3.RK443, d3.SBDF2), (d3.RK443, d3.SBDF3), (d3.RK443, d3.SBDF4)]
# timesteppers = [(d3.RK443, d3.SBDF2), (d3.RK443, d3.MCNAB2), (d3.RK443, d3.CNLF2), (d3.RK443, d3.CNAB2)]
# timesteppers = [(d3.RK443, d3.SBDF2), (d3.RK443, d3.RK222), (d3.RK443, d3.MCNAB2)]
# timesteppers = [(d3.RK443, d3.RK111), (d3.RK443, d3.RK222), (d3.RK443, d3.RK443), (d3.RK443, d3.RKGFY), (d3.RK443, d3.RKSMR)]

def compute_objective(guess):
    kx = 1
    timestepper_pair = timesteppers[0]
    forward_solver = forward_problem.build_solver(timestepper_pair[0])
    backward_solver = backward_problem.build_solver(timestepper_pair[1])

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

    # guess_data = np.loadtxt(path + '/diffusion_guess.txt')
    opt.ic['u']['g'] = guess
    # opt.ic['u']['g'] = 0.0

    U_data = np.loadtxt(path + '/diffusion_U.txt')
    u = next(field for field in forward_solver.state if field.name == 'u')
    U = dist.Field(name='U', bases=xbasis)
    U['g'] = U_data


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
        gamma = 0.0
        jac = kwargs['jac']
        for i in range(maxiter):
            old_f = f
            f = fun(x0)
            gradf = jac(x0)
            old_gamma = gamma
            gamma = opt.compute_gamma(1.0)
            if i > 0:
                step_p = (old_f - f) / old_gamma / opt.old_grad_sqrd
                # logger.info("step_p = {}".format(step_p))
                opt.metricsT_norms['step_p'] = step_p

            x0 -= gamma * gradf
        return optimize.OptimizeResult(x=x0, success=True, message='beep boop')

    method = 'L-BFGS-B'
    if (method == "euler"):
        method = euler_descent

    startTime = datetime.now()

    x0 = opt.ic['u']['g'].flatten().copy()  # Initial guess.
    obj = opt.loop_forward(x0)
    opt.loop_backward(x0)
    grad = opt.new_grad['g'].copy()
    return obj, grad

    # try:
    #     tol = 1e-10
    #     options = {'maxiter' : opt_iters, 'ftol' : tol, 'gtol' : tol}
    #     x0 = opt.ic['u']['g'].flatten().copy()  # Initial guess.
    #     res1 = optimize.minimize(opt.loop_forward, x0, jac=opt.loop_backward, method=method, tol=tol, options=options)
    #     # res1 = optimize.minimize(opt.loop_forward, x0, jac=opt.loop_backward, method='L-BFGS-B', tol=tol, options=options)
    #     # res1 = optimize.minimize(opt.loop_forwaoh rd, x0, jac=opt.loop_backward, method=euler_descent, options=options)
    #     logger.info('scipy message {}'.format(res1.message))

    # except opt.LoopIndexException as e:
    #     details = e.args[0]
    #     logger.info(details["message"])
    # except opt.NanNormException as e:
    #     details = e.args[0]
    #     logger.info(details["message"])
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
    grads.append(opt.new_grad['g'].copy())

    mag_ap_grad = np.max(np.abs((soln - opt.ic['u']['g'])[0, :]))
    apgrad = -(soln - opt.ic['u']['g'])[0, :] / mag_ap_grad
    numgrad = grads[-1] / eps

    angle = np.arccos(np.inner(apgrad, numgrad) / (norm(apgrad, ord=1) * norm(numgrad, ord=1)))
    angles.append(angle * 180 / np.pi)



n = 20
mu = 5.5
sig = 0.5
soln = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
# delta = -eps*np.exp(-np.power(x - 4.0, 2.) / (2 * np.power(sig, 2.)))
delta = eps*np.sin((8) * x*np.pi / Lx)
guess = soln + delta


L_guess, grad = compute_objective(guess)

plt.plot(x, grad)
plt.plot(x,delta)
plt.show()
sys.exit()

# delL_ap = []
# delL_diff = []

# delta /= np.max(np.abs(delta))
# grad /= np.max(np.abs(grad))

delta /= np.linalg.norm(delta)
grad /= np.linalg.norm(grad)

varepss = np.linspace(0.0, 4*eps, 30)
for vareps in varepss:
    logger.info('VAREPS = {}'.format(vareps))
    delL_ap.append(compute_objective(guess  - vareps * delta)[0] - L_guess)
    delL_diff.append(compute_objective(guess  - vareps * grad)[0] - L_guess)

plt.scatter(varepss, np.array(delL_diff) - np.array(delL_ap), label = 'Apriori Gradient')
# plt.scatter(varepss, , label = 'Diffused Gradient')
# plt.legend()
plt.ylabel(r'$\delta \mathcal{L}_{diff} - \delta \mathcal{L}_{apriori}$')
plt.xlabel(r'$\varepsilon$')
plt.title(r'$\epsilon = $' + str(eps) + r'; $L2$ Normalization')
plt.savefig(path + '/deltaL_L2_comp.png')
plt.show()

# plt.plot(x, grad)
# plt.plot(x, delta)
# plt.show()

# import matplotlib as mpl
# mpl.rcParams['lines.linewidth'] = 2

# from itertools import cycle
# lines = ["-","--","-.",":"]
# linecycler = cycle(lines)



# print(angles)

# plt.scatter(wavenumbers, angles)
# plt.xlabel('wavenumbers')
# plt.show()

# plt.plot(x, -(soln - opt.ic['u']['g'])[0, :] / mag_ap_grad, label='(-) apriori gradient', linewidth=4, color='black')
# plt.plot(x, delta / eps)
# plt.show()
