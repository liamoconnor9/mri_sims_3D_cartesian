"""
Usage:
    plot_paths.py <config_file>
"""
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

args = docopt(__doc__)
try:
    filename = Path(args['<config_file>'])
except:
    filename = Path('kdv_options.cfg')
    
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
euler_safety = config.getfloat('parameters', 'euler_safety')
gamma_init = config.getfloat('parameters', 'gamma_init')

periodic = config.getboolean('parameters', 'periodic')
show_forward = config.getboolean('parameters', 'show')
show_iter_cadence = config.getint('parameters', 'show_iter_cadence')
show_loop_cadence = config.getint('parameters', 'show_loop_cadence')

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

write_suffix = str(config.get('parameters', 'suffix'))

soln = 2*np.sin(2*np.pi*x / Lx) + 2*np.sin(4*np.pi*x / Lx)
# soln = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
soln_f = dist.Field(name='soln_f', bases=xbasis)
soln_f['g'] = soln.reshape((1, N))

guess = 0.0


path = os.path.dirname(os.path.abspath(__file__))
U_data = np.loadtxt(path + '/kdv_U.txt')
u = next(field for field in forward_solver.state if field.name == 'u')

mode1 = dist.Field(name='mode1', bases=xbasis)
mode1['g'] = np.sin(2*np.pi*x / Lx)

mode2 = dist.Field(name='mode2', bases=xbasis)
mode2['g'] = np.sin(4*np.pi*x / Lx)

import glob
Nics = len(glob.glob1(path + '/' + write_suffix, "*.pick"))
print('number of procs = {}'.format(Nics))

for i in range(Nics):
    tracker_name = path + '/' + write_suffix + '/tracker_rank' + str(i) + '.pick'
    with open(tracker_name, 'rb') as file:
        tracker = pickle.load(file)
    x = tracker['x']
    objs = tracker['objectiveT']
    c1 = []
    c2 = []
    logger.info('plotting {}/{}'.format(i, Nics))
    print(len(x))
    for ptind in range(len(x)):
        x0 = x[ptind]
        u.change_scales(1)
        u['g'] = x0.copy()
        c1.append(d3.Integrate(u*mode1).evaluate()['g'].flat[0] / Lx * 2.0)
        c2.append(d3.Integrate(u*mode2).evaluate()['g'].flat[0] / Lx * 2.0)
    pc = plt.scatter(c1, c2, c=objs, cmap='seismic')
plt.colorbar(pc)
plt.xlabel(r'$(2/L_x) \; \langle u\sin${}$x \rangle$'.format(1))
plt.ylabel(r'$(2/L_x) \; \langle u\sin${}$x \rangle$'.format(2))
plt.title(r'$\langle (u - U)^2 \rangle$; a = {}, b = {}, T = {}'.format(a, b, T))
plt.savefig(path + '/' + write_suffix + '/paths_2d_long.png')
logger.info('saved fig to directory: {}'.format(path + '/' + write_suffix + '/paths_2d_long.png'))
