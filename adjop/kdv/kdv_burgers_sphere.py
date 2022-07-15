"""
Dedalus script simulating the 1D Korteweg-de Vries / Burgers equation.
This script demonstrates solving a 1D initial value problem and produces
a space-time plot of the solution. It should take just a few seconds to
run (serial only).
We use a Fourier basis to solve the IVP:
    dt(u) + u*dx(u) = a*dx(dx(u)) + b*dx(dx(dx(u)))
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import os
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from mpi4py import MPI
CW = MPI.COMM_WORLD

filename = Path('kdv_options.cfg')
config = ConfigParser()
config.read(str(filename))

logger.info('Running kdv_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
write_suffix = str(config.get('parameters', 'suffix'))
Lx = config.getfloat('parameters', 'Lx')
Nx = config.getint('parameters', 'Nx')

a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')
sig = config.getfloat('parameters', 'sig_ic')
mu = config.getfloat('parameters', 'mu_ic')
ic_scale = config.getfloat('parameters', 'ic_scale')
stop_sim_time = config.getfloat('parameters', 'T')
timestep = config.getfloat('parameters', 'dt')

R = config.getfloat('parameters', 'R')
modes_dim = config.getint('parameters', 'modes_dim')

# Simulation Parameters
dealias = 3/2
dtype = np.float64

periodic = config.getboolean('parameters', 'periodic')
show = config.getboolean('parameters', 'show')
show_iter_cadence = config.getint('parameters', 'show_iter_cadence')

timestepper = d3.RK443
epsilon_safety = 1

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype, comm=MPI.COMM_SELF)

if (periodic):
    xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
    u = dist.Field(name='u', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)

else:
    xbasis = d3.ChebyshevT(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
    u = dist.Field(name='u', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)
    
    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    tau_3 = dist.Field(name='tau_3')
    lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    ux = dx(u) + lift(tau_1) # First-order reduction
    uxx = dx(ux) + lift(tau_2) # First-order reduction

vars = [u]
if (not periodic):
    vars.append(tau_1)
    vars.append(tau_2)
    vars.append(tau_3)

# Problem
problem = d3.IVP(vars, namespace=locals())
if periodic:
    problem.add_equation("dt(u) - a*dx(dx(u)) - b*dx(dx(dx(u))) = - u*dx(u)")
else:
    problem.add_equation("dt(u) - a*uxx - b*dx(uxx) + lift(tau_3) = - u*ux")
    problem.add_equation("u(x='left') = 0")
    problem.add_equation("u(x='right') = 0")
    problem.add_equation("ux(x='left') = 0")

# Initial conditions
x = dist.local_grid(xbasis)
u['g'] = ic_scale*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

np.random.seed(CW.rank)
coeffs = 2*np.random.rand(modes_dim) - 1.0
coeffs *= R / np.sqrt(np.sum(coeffs**2))
for kx, coeff in enumerate(coeffs):
    u['g'] += coeff*np.cos((kx + 1)*2*np.pi*x / Lx)

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Main loop
# Main loop
u.change_scales(1)
if (show):
    fig = plt.figure()
    p, = plt.plot(x, u['g'])
    fig.canvas.draw()
    title = plt.title('t=%f' %solver.sim_time)

udata = []
times = []
u.change_scales(1)
udata.append(u['g'].copy())
times.append(solver.sim_time)

for iter in range(int(solver.stop_sim_time // timestep) + 1):
    solver.step(timestep)
    if solver.iteration % 100 == 0:
        logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    if show and solver.iteration % show_iter_cadence == 0:
        u.change_scales(1)
        p.set_ydata(u['g'])
        plt.title('t=%f' %solver.sim_time)
        plt.pause(1e-10)
        fig.canvas.draw()
    if solver.iteration % show_iter_cadence == 0:
        u.change_scales(1)
        udata.append(u['g'].copy())
        times.append(solver.sim_time)


logger.info('solve complete, sim time = {}'.format(solver.sim_time))
u.change_scales(1)
u_T = u['g'].copy()
path = os.path.dirname(os.path.abspath(__file__))

udata = np.array(udata)
times = np.array(times)
pc = plt.pcolormesh(x.ravel(), times.ravel(), udata, cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
plt.colorbar(pc)
plt.xlim(0, Lx)
plt.ylim(0, times[-1])
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.title('Forward KdV: Initial Guess (Loop Index 0)')
plt.savefig(path + '/' + write_suffix + '/loopind{}.png'.format(CW.rank))
logger.info('figure saved to ' + path + '/' + write_suffix + '/loopind{}.png'.format(CW.rank))
CW.Barrier()

if(np.isnan(udata).any()):
    print('Initial guess simulation failed! rank = {}'.format(CW.rank))
    raise