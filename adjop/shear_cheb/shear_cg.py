"""
Dedalus script for adjoint looping:
Given an end state (checkpoint_U), this script recovers the initial condition with no prior knowledge
Usage:
    shear_cg.py <config_file> <run_suffix>
"""

from distutils.command.bdist import show_formats
import os
import pickle
path = os.path.dirname(os.path.abspath(__file__))
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import sys
sys.path.append(path + "/..")
import h5py
import gc
import dedalus.public as d3
from dedalus.core import domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
logging.getLogger('solvers').setLevel(logging.ERROR)
# logger.setLevel(logging.info)
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser

from OptimizationContext import OptimizationContext
from ShearOptimization import ShearOptimization
import ForwardShear
import BackwardShear
import PressureBVP
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping, OptimizeResult
from natsort import natsorted

args = docopt(__doc__)
filename = Path(args['<config_file>'])
write_suffix = args['<run_suffix>']

config = ConfigParser()
config.read(str(filename))
path + '/' + write_suffix
# fh = logging.FileHandler(path + '/' + write_suffix + '/log.log')
# fh.setLevel(logging.INFO)
# logger.addHandler(fh)

logger.info('Running shear_flow.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
load_state = config.getboolean('parameters', 'load_state')
metrics_fn = str(config.get('parameters', 'metric_fn'))
basinhopping_iters = config.getint('parameters', 'basinhopping_iters')
opt_cycles = config.getint('parameters', 'opt_cycles')
opt_iters = config.getint('parameters', 'opt_iters')
method = str(config.get('parameters', 'scipy_method'))
euler_safety = config.getfloat('parameters', 'euler_safety')
gamma_init = config.getfloat('parameters', 'gamma_init')

opt_scales = config.getfloat('parameters', 'opt_scales')
opt_layout = str(config.get('parameters', 'opt_layout'))

num_cp = config.getint('parameters', 'num_cp')
handler_loop_cadence = config.getint('parameters', 'handler_loop_cadence')
add_handlers = config.getboolean('parameters', 'add_handlers')
guide_coeff = config.getfloat('parameters', 'guide_coeff')

u_weight = config.getfloat('parameters', 'u_weight')
omega_weight = config.getfloat('parameters', 'omega_weight')
s_weight = config.getfloat('parameters', 's_weight')
psi_weight = config.getfloat('parameters', 'psi_weight')
cc_weight = config.getfloat('parameters', 'cc_weight')

Lx = config.getfloat('parameters', 'Lx')
Lz = config.getfloat('parameters', 'Lz')
Nx = config.getint('parameters', 'Nx')
Nz = config.getint('parameters', 'Nz')

Reynolds = config.getfloat('parameters', 'Re')
T = config.getfloat('parameters', 'T')
dt = config.getfloat('parameters', 'dt')

dealias = 3/2
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
coords.name = coords.names

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
x, z = dist.local_grids(bases[0], bases[1])
ex, ez = coords.unit_vector_fields(dist)

domain = domain.Domain(dist, bases)
dist = domain.dist

forward_problem = ForwardShear.build_problem(domain, coords, Reynolds)
backward_problem = BackwardShear.build_problem(domain, coords, Reynolds)

# forward, and corresponding adjoint variables (fields)
u = forward_problem.variables[0]
s = forward_problem.variables[1]
p = forward_problem.variables[2]
psi = forward_problem.variables[3]

u_t = backward_problem.variables[0]
s_t = backward_problem.variables[1]

lagrangian_dict = {u : u_t, s : s_t}

forward_solver = forward_problem.build_solver(d3.RK443)
backward_solver = backward_problem.build_solver(d3.RK222)

opt = ShearOptimization(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)
opt.opt_iters = opt_iters
opt.add_handlers = add_handlers
opt.handler_loop_cadence = handler_loop_cadence

U = dist.VectorField(coords, name='U', bases=bases)
S = dist.Field(name='S', bases=bases)
P = dist.Field(name='P', bases=bases)
Psi = dist.Field(name='Psi', bases=bases)

X = dist.Field(name='X', bases=bases)
Z = dist.Field(name='Z', bases=bases)
X['g'] = x - Lx/2.0
Z['g'] = z

slices = dist.grid_layout.slices(domain, scales=1)
opt.slices = slices
opt.metrics_fn = metrics_fn

# Populate U with end state of known initial condition
end_state_path = path + '/' + write_suffix + '/checkpoint_target/checkpoint_target_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
    S['g'] = f['tasks/s'][-1, :, :][slices[0], slices[1]]
    P['g'] = f['tasks/p'][-1, :, :][slices[0], slices[1]]
    Psi['g'] = f['tasks/psi'][-1, :, :][slices[0], slices[1]]
    logger.info('loading target {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))

restart_dir = path + '/' + write_suffix + '/checkpoints'
if (load_state and len(os.listdir(restart_dir)) <= 1):
    logger.info('No checkpoints found in {}! Restarting... '.format(restart_dir))
    load_state = False

if (load_state):
    checkpoint_names = [name for name in os.listdir(restart_dir) if 'loop' in name]
    last_checkpoint = natsorted(checkpoint_names)[-1]
    restart_file = restart_dir + '/' + last_checkpoint + '/' + last_checkpoint + '_s1.h5'
    with h5py.File(restart_file) as f:
        opt.ic['u']['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
        S['g'] = f['tasks/s'][-1, :, :][slices[0], slices[1]]
        logger.info('loading loop {}'.format(restart_file))
        loop_str_index = restart_file.rfind('loop') + 4
        opt.loop_index = int(restart_file[loop_str_index:-6])
        with open(path + '/' + write_suffix + '/' + metrics_fn, 'rb') as f:
            opt.metricsT_norms_lists = pickle.load(f)
            truncate = max([len(metric_list) for metric_list in opt.metricsT_norms_lists.values()]) - opt.loop_index
            for metric_list in opt.metricsT_norms_lists.values():
                del metric_list[-truncate:]

else:
    # opt.ic['u'].fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise

    # opt.ic['u']['g'][0] = guide_coeff * (1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1)))
    # opt.ic['u']['g'][1] += guide_coeff * 0.1 * np.sin(2*np.pi*x/Lx)
    # opt.ic['u']['g'][1] += guide_coeff * 0.1 * np.sin(2*np.pi*x/Lx)

    #old ic
    sigma = 0.15
    opt.ic['u']['g'][0] = guide_coeff * 1/2 * np.tanh((z)/0.1)
    opt.ic['u']['g'][0] += guide_coeff * (z) * np.exp(-0.5 * ( (x - 0.5)**2 / sigma**2 + (z)**2 / sigma**2) )
    opt.ic['u']['g'][1] += guide_coeff * -(x - 0.5) * np.exp(-0.5 * ( (x - 0.5)**2 / sigma**2 + (z)**2 / sigma**2) )

# PressureBVP.build_problem(domain, coords, P)
# sys.exit()

# set tracer initial condition
opt.ic['s'] = dist.Field(name='s', bases=bases)
opt.ic['s']['g'] = 1/2 * np.tanh((z)/0.1)

# opt.backward_ic['s_t'] = dist.Field(name='s_t', bases=bases)
# opt.backward_ic['s_t']['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))

# Late time objective: objectiveT is minimized at t = T
# w2 = d3.div(d3.skew(u))
dx = lambda A: d3.Differentiate(A, coords['x'])
dz = lambda A: d3.Differentiate(A, coords['z'])
curl = lambda A: d3.skew(d3.grad(A))
lap  = lambda A: d3.div(d3.grad(A))
integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')

ux = u @ ex
uz = u @ ez
w = dx(uz) - dz(ux)

Ux = U @ ex
Uz = U @ ez
W = dx(Uz) - dz(Ux)

objectiveT = d3.dot(u - U, u - U)
opt.set_objectiveT(objectiveT)
opt.objectiveT *= (u_weight + cc_weight)
opt.backward_ic['u_t'] *= u_weight

opt.objectiveT += omega_weight * (w - W)**2
opt.objectiveT += s_weight     * (s - S)**2
opt.objectiveT += psi_weight   * (psi - Psi)**2

opt.backward_ic['s_t']  =  2*s_weight*(s - S)
opt.backward_ic['u_t'] += -2.0*omega_weight*d3.skew(d3.grad((w - W)))
opt.backward_ic['u_t'] +=  2.0*psi_weight* ( (psi - Psi).evaluate().antidifferentiate(zbasis, ('left', 0)) )
# opt.backward_ic['u_t'] +=  2.0*psi_weight* ( (psi - Psi) * (Z*ex - X*ez) - integ((psi - Psi) * (Z*ex - X*ez)) )
opt.backward_ic['u_t'] +=  2.0*cc_weight*curl(lap(w - W))

# opt.backward_ic['u_t'] += 2.0*psi_weight*d3.skew(d3.grad((psi - Psi)))

opt.metricsT['u_error'] = d3.dot(u - U, u - U)
opt.metricsT['omega_error'] = (w - W)**2
opt.metricsT['s_error'] = (s - S)**2
opt.metricsT['p_error'] = (p - P)**2
opt.metricsT['psi_error'] = (psi - Psi)**2

opt.track_metrics()

def check_status(x, f, accept):
    logger.info('jumping..')
    CW.barrier()

def euler_descent(fun, x0, args, **kwargs):
    # gamma = 0.001
    # maxiter = kwargs['maxiter']
    maxiter = opt_iters
    jac = kwargs['jac']
    f = np.nan
    gamma = 0.0000000025
    for i in range(opt.loop_index, maxiter):
        old_f = f
        f, gradf = opt.loop(x0)
        # gradf /= opt.new_grad_sqrd**(0.5)
        # gamma = 0.1 * f

        old_gamma = gamma
        if i > 0:
            opt.compute_gamma(euler_safety)
            step_p = (old_f - f) / old_gamma / (opt.old_grad_sqrd)
            opt.metricsT_norms['step_p'] = step_p
        # else:
        #     gamma = gamma_init
        opt.metricsT_norms['gamma'] = gamma
        x0 -= 1e4 * gamma * gradf
    logger.info('success')
    logger.info('maxiter = {}'.format(maxiter))
    return OptimizeResult(x=x0, success=True, message='beep boop')

if (method == "euler"):
    method = euler_descent

# logging.basicConfig(filename='/path/to/your/log', level=....)
# logging.basicConfig(filename = opt.run_dir + '/' + opt.write_suffix + '/log.txt')

from datetime import datetime
startTime = datetime.now()

# Parameters to choose how in what dedalus layout scipy will optimize: e.g. optimize in grid space or coeff with some scale
opt.opt_scales = opt_scales
opt.opt_layout = opt_layout
opt.dist_layout = dist.layout_references[opt_layout]
opt.opt_slices = opt.dist_layout.slices(domain, scales=opt_scales)

opt.ic['u'].change_scales(opt_scales)
opt.ic['u'][opt_layout]


options = {'maxiter' : opt_iters}
minimizer_kwargs = {"method":method, "jac":True}
if (basinhopping_iters > 0):
    try:
        x0 = opt.ic['u'].allgather_data(layout=opt.dist_layout).flatten().copy()  # Initial guess.
        res1 = basinhopping(opt.loop, x0, T=1e-2, callback=check_status, minimizer_kwargs=minimizer_kwargs)
        # res1 = basinhopping(opt.loop, x0, T=0.1, niter=basinhopping_iters, callback=check_status, minimizer_kwargs=minimizer_kwargs)
        logger.info(res1)
    except opt.LoopIndexException as e:
        details = e.args[0]
        logger.info(details["message"])
    except opt.NanNormException as e:
        details = e.args[0]
        logger.info(details["message"])
else:
    for cycle_ind in range(opt_cycles):
        logger.info('Initiating optimization cycle {}'.format(cycle_ind))
        try:
            x0 = opt.ic['u'].allgather_data(layout=opt.dist_layout).flatten().copy()  # Initial guess.
            res1 = minimize(opt.loop_forward, x0, jac=opt.loop_backward, options=options, tol=1e-8, method=method)
            logger.info(res1)
        except opt.LoopIndexException as e:
            details = e.args[0]
            logger.info(details["message"])
        except opt.NanNormException as e:
            details = e.args[0]
            logger.info(details["message"])
        except Exception as e:
            logger.info('Unknown exception occured: {}'.format(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.info(exc_type, fname, exc_tb.tb_lineno)
        opt.opt_iters += opt_iters

logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
logger.info('BEST LOOP INDEX {}'.format(opt.best_index))
logger.info('BEST objectiveT {}'.format(opt.best_objectiveT))
logger.info('####################################################')

# for metricT_name in opt.metricsT_norms_lists.keys():
    # logger.(opt.metricsT_norms_lists[metricT_name])

if CW.rank == 0:

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    ax1.plot(opt.indices, opt.objectiveT_norms)
    ax1.set_xlabel('loop index')
    ax1.set_ylabel('ObjectiveT')
    ax1.set_yscale('log')
    # ax1.title.set_text(str(opt.objectiveT))

    keys = list(opt.metricsT_norms_lists.keys())
    ax2.plot(opt.indices, opt.metricsT_norms_lists[keys[0]])
    ax2.set_xlabel('loop index')
    ax2.set_ylabel(keys[0])
    ax2.set_yscale('log')
    # ax2.title.set_text(str(opt.metricsT[keys[0]]))

    ax3.plot(opt.indices, opt.metricsT_norms_lists[keys[1]])
    ax3.set_xlabel('loop index')
    ax3.set_ylabel(keys[1])
    ax3.set_yscale('log')
    # ax3.title.set_text(str(opt.metricsT[keys[1]]))

    fig.suptitle(write_suffix)
    plt.savefig(opt.run_dir + '/' + opt.write_suffix + '/metricsT.png')
    logger.info('metricsT fig saved')