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
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime

class KdvOptimization(OptimizationContext):
    def before_fullforward_solve(self):
        if (self.loop_index % self.show_loop_cadence == 0 and self.show):
            u = self.forward_solver.state[0]
            u.change_scales(1)
            self.fig = plt.figure()
            self.p, = plt.plot(self.x_grid, u['g'])
            plt.plot(self.x_grid, self.U_data)
            self.fig.canvas.draw()
            title = plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.forward_solver.sim_time, 1)))
            # plt.show()
                
        logger.debug('start evaluating f..')

    def during_fullforward_solve(self):
        if (self.loop_index % self.show_loop_cadence == 0 and self.show and self.forward_solver.iteration % self.show_iter_cadence == 0):
            u = self.forward_solver.state[0]
            u.change_scales(1)
            self.p.set_ydata(u['g'])
            plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.forward_solver.sim_time, 1)))
            plt.pause(1e-10)
            self.fig.canvas.draw()

    def after_fullforward_solve(self):
        loop_message = 'loop index = {}; '.format(self.loop_index)
        loop_message += 'objectiveT = {}; '.format(self.objectiveT_norm)
        for metric_name in self.metricsT_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, self.metricsT_norms[metric_name])
        logger.info(loop_message)
        plt.pause(3e-1)
        plt.close()

    def before_backward_solve(self):
        logger.debug('Starting backward solve')
        if (self.loop_index % self.show_loop_cadence == 0 and self.show):
            u = self.backward_solver.state[0]
            u.change_scales(1)
            self.fig = plt.figure()
            self.p, = plt.plot(self.x_grid, u['g'])
            self.fig.canvas.draw()
            title = plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.backward_solver.sim_time, 1)))

    def during_backward_solve(self):
        if (self.loop_index % self.show_loop_cadence == 0 and self.show and self.backward_solver.iteration % self.show_iter_cadence == 0):
            u = self.backward_solver.state[0]
            u.change_scales(1)
            self.p.set_ydata(u['g'])
            plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.backward_solver.sim_time, 1)))
            plt.pause(1e-10)
            self.fig.canvas.draw()
        # logger.debug('backward solver time = {}'.format(self.backward_solver.sim_time))
        pass

    def after_backward_solve(self):
        plt.pause(3e-1)
        plt.close()
        logger.debug('Completed backward solve')
        pass

filename = path + '/kdv_options.cfg'
config = ConfigParser()
config.read(str(filename))

logger.info('Running kdv_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
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
forward_problem = ForwardKDV.build_problem(domain, xcoord, a, b)
backward_problem = BackwardKDV.build_problem(domain, xcoord, a, b)

# Names of the forward, and corresponding adjoint variables
lagrangian_dict = {forward_problem.variables[0] : backward_problem.variables[0]}
grads = []

timesteppers = [(d3.RK443, d3.SBDF1), (d3.RK443, d3.SBDF2), (d3.RK443, d3.SBDF3)]
# timesteppers = [(d3.RK443, d3.SBDF1), (d3.RK443, d3.SBDF2), (d3.RK443, d3.SBDF3), (d3.RK443, d3.SBDF4)]
# timesteppers = [(d3.RK443, d3.SBDF2), (d3.RK443, d3.MCNAB2), (d3.RK443, d3.CNLF2), (d3.RK443, d3.CNAB2)]
# timesteppers = [(d3.RK443, d3.SBDF2), (d3.RK443, d3.RK222), (d3.RK443, d3.MCNAB2)]
timesteppers = [(d3.RK443, d3.SBDF4), (d3.RK443, d3.RK443)]
# timesteppers = [(d3.RK443, d3.RK222)]
# timesteppers = [(d3.RK443, d3.RK111), (d3.RK443, d3.RK222), (d3.RK443, d3.RK443), (d3.RK443, d3.RKGFY), (d3.RK443, d3.RKSMR)]
for timestepper_pair in timesteppers:
    
    forward_solver = forward_problem.build_solver(timestepper_pair[0])
    backward_solver = backward_problem.build_solver(timestepper_pair[1])

    write_suffix = 'kdv0'

    opt = KdvOptimization(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
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

    n = 20
    mu = 4.1
    sig = 0.5
    guess = -np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    guess = x*0
    # delta = -0.0*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    delta = -0.0001*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    guess = soln + delta

    guess_data = np.loadtxt(path + '/kdv_guess.txt')
    opt.ic['u']['g'] = guess_data
    # opt.ic['u']['g'] = 0.0

    U_data = np.loadtxt(path + '/kdv_U.txt')
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
    grads.append(opt.new_grad['g'].copy())

plt.plot(x, -(soln - opt.ic['u']['g'])[0, :], label='apriori gradient')
mag_ap_grad = np.max(np.abs((soln - opt.ic['u']['g'])[0, :]))
for i in range(len(timesteppers)):
    grad = grads[i]
    normed_grad = grad * mag_ap_grad / np.max(np.abs(grad))
    plt.plot(x, normed_grad, label='{}, {}'.format(timesteppers[i][0].__name__, timesteppers[i][1].__name__), linestyle=':')

plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\mu(x, 0)$')
plt.title('Gradient')
plt.savefig(path + '/grad_u.png')
plt.show()

# u = opt.ic['u']
# u.change_scales(1)
# u_T = u['g'].copy()
# path = os.path.dirname(os.path.abspath(__file__))
# np.savetxt(path + '/kdv_guess.txt', u_T)
# logger.info('saved final state')

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
