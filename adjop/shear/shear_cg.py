"""
Dedalus script for adjoint looping:
Given an end state (checkpoint_U), this script recovers the initial condition with no prior knowledge
Usage:
    shear_cg.py <config_file> <run_suffix>
"""

from distutils.command.bdist import show_formats
import os
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
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser

from OptimizationContext import OptimizationContext
import ForwardShear
import BackwardShear
import matplotlib.pyplot as plt
from scipy import optimize

class ShearOptimization(OptimizationContext):
    def before_fullforward_solve(self):
        if self.add_handlers:
            snapshots = self.forward_solver.evaluator.add_file_handler(self.run_dir + '/' + self.write_suffix + '/snapshots_forward/snapshots_forward_loop' + str(self.loop_index), sim_dt=0.01, max_writes=10, mode='overwrite')
            u = self.forward_solver.state[0]
            s = self.forward_solver.state[1]
            p = self.forward_solver.state[2]
            snapshots.add_task(s, name='tracer')
            snapshots.add_task(p, name='pressure')
            snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
            snapshots.add_task(d3.dot(ex, u), name='ux')
            snapshots.add_task(d3.dot(ez, u), name='uz')

        logger.info('start evaluating f..')

    def during_fullforward_solve(self):
        pass

    def after_fullforward_solve(self):
        logger.info('done evaluating f. norm = {}'.format(self.ObjectiveT_norm))

    def before_backward_solve(self):
        if self.add_handlers:
            # setting tracer to end state of forward solve
            self.forward_solver.state[1].change_scales(1)
            self.backward_solver.state[1]['g'] = self.forward_solver.state[1]['g'].copy() - self.S['g'].copy()
            snapshots_backward = self.backward_solver.evaluator.add_file_handler(self.run_dir + '/' + self.write_suffix + '/snapshots_backward/snapshots_backward_loop' + str(self.loop_index), sim_dt=-0.01, max_writes=10, mode='overwrite')
            u_t = self.backward_solver.state[0]
            s_t = self.backward_solver.state[1]
            p_t = self.backward_solver.state[2]
            snapshots_backward.add_task(s_t, name='tracer')
            snapshots_backward.add_task(p_t, name='pressure')
            snapshots_backward.add_task(-d3.div(d3.skew(u_t)), name='vorticity')
            snapshots_backward.add_task(d3.dot(ex, u_t), name='ux')
            snapshots_backward.add_task(d3.dot(ez, u_t), name='uz')
        logger.debug('Starting backward solve')

    def during_backward_solve(self):
        # logger.debug('backward solver time = {}'.format(self.backward_solver.sim_time))
        pass

    def after_backward_solve(self):
        logger.info('Completed backward solve')
        pass

args = docopt(__doc__)
filename = Path(args['<config_file>'])
write_suffix = args['<run_suffix>']

config = ConfigParser()
config.read(str(filename))

logger.info('Running shear_flow.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
opt_iters = config.getint('parameters', 'opt_iters')
num_cp = config.getint('parameters', 'num_cp')
add_handlers = config.getboolean('parameters', 'add_handlers')

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
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
x, z = dist.local_grids(bases[0], bases[1])
ex, ez = coords.unit_vector_fields(dist)

domain = domain.Domain(dist, bases)
dist = domain.dist

forward_problem = ForwardShear.build_problem(domain, coords, Reynolds)
backward_problem = BackwardShear.build_problem(domain, coords, Reynolds)

# forward, and corresponding adjoint variables (fields)
u = forward_problem.variables[0]
u_t = backward_problem.variables[0]
lagrangian_dict = {u : u_t}

forward_solver = forward_problem.build_solver(d3.RK443)
backward_solver = backward_problem.build_solver(d3.SBDF4)

opt = ShearOptimization(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
opt.set_time_domain(T, num_cp, dt)
opt.opt_iters = opt_iters
opt.add_handlers = add_handlers

U = dist.VectorField(coords, name='U', bases=bases)
S = dist.Field(name='S', bases=bases)
slices = dist.grid_layout.slices(domain, scales=1)

# Populate U with end state of known initial condition
end_state_path = path + '/' + write_suffix + '/checkpoint_target/checkpoint_target_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices[0], slices[1]]
    S['g'] = f['tasks/s'][-1, :, :][slices[0], slices[1]]
    logger.info('looding end state {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))

opt.S = S
# opt.ic['u'].fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
coeff = 0.0
opt.ic['u']['g'][0] = coeff * (1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1)))
opt.ic['u']['g'][1] += coeff * (0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01))
opt.ic['u']['g'][1] += coeff * (0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01))

opt.ic['s'] = dist.Field(name='s', bases=bases)
opt.ic['s']['g'] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))

# Late time objective: ObjectiveT is minimized at t = T
w2 = d3.div(d3.skew(u))
W2 = d3.div(d3.skew(U))
# ObjectiveT = 0.5*(w2 - W2)**2

ObjectiveT = 0.5*d3.dot(u - U, u - U)
opt.set_objectiveT(ObjectiveT)
# opt.backward_ic['u_t'] = (u - U)

from datetime import datetime
startTime = datetime.now()

opt.ic['u'].change_scales(1)
opt.ic['u']['g']
x0 = opt.ic['u'].allgather_data().flatten().copy()  # Initial guess.

def check_status(x):
    logger.info('completed scipy py iteration')
    CW.barrier()

try:
    res1 = optimize.fmin_cg(opt.loop_forward, x0, fprime=opt.loop_backward, callback=check_status)
except opt.LoopIndexException as e:
    details = e.args[0]
    logger.info(details["message"])
except opt.NanNormException as e:
    details = e.args[0]
    logger.info(details["message"])
except Exception as e:
    logger.info('Unknown exception occured: {}'.format(e))

logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
logger.info('BEST LOOP INDEX {}'.format(opt.best_index))
logger.info('BEST ObjectiveT {}'.format(opt.best_objectiveT))
logger.info('####################################################')
