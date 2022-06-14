"""
Dedalus script simulating a 2D periodic incompressible shear flow with a passive
tracer field for visualization. This script demonstrates solving a 2D periodic
initial value problem. It can be ran serially or in parallel, and uses the
built-in analysis framework to save data snapshots to HDF5 files. The
`plot_snapshots.py` script can be used to produce plots from the saved data.
The simulation should take a few cpu-minutes to run.

The initial flow is in the x-direction and depends only on z. The problem is
non-dimensionalized usign the shear-layer spacing and velocity jump, so the
resulting viscosity and tracer diffusivity are related to the Reynolds and
Schmidt numbers as:

    nu = 1 / Reynolds
    D = nu / Schmidt

Usage:
    shear_flow.py <config_file> <run_suffix>
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import os
path = os.path.dirname(os.path.abspath(__file__))
from mpi4py import MPI
CW = MPI.COMM_WORLD
import sys
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser

# if len(sys.argv) > 1:
#     suffix = sys.argv[1]
# else:
#     suffix = ''

args = docopt(__doc__)
filename = Path(args['<config_file>'])
suffix = args['<run_suffix>']

config = ConfigParser()
config.read(str(filename))

logger.info('Running shear_flow.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
Lx = config.getfloat('parameters', 'Lx')
Lz = config.getfloat('parameters', 'Lz')
Nx = config.getint('parameters', 'Nx')
Nz = config.getint('parameters', 'Nz')

Reynolds = config.getfloat('parameters', 'Re')
stop_sim_time = config.getfloat('parameters', 'T')
max_timestep = config.getfloat('parameters', 'dt') 
add_handlers = config.getboolean('parameters', 'add_handlers')

nu = 1 / Reynolds
Schmidt = 0.1
D = nu / Schmidt
dealias = 3/2
timestepper = d3.RK443
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
s = dist.Field(name='s', bases=(xbasis, zbasis))

tau_p = dist.Field(name='tau_p')
tau1u = dist.VectorField(coords, name='tau1u', bases=(xbasis))
tau2u = dist.VectorField(coords, name='tau2u', bases=(xbasis))
tau1s = dist.Field(name='tau1s', bases=(xbasis))
tau2s = dist.Field(name='tau2s', bases=(xbasis))

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')

# nccs
S = 1
# U0 = dist.VectorField(coords, name='U0', bases=zbasis)
# U0['g'][0] = S * z

lift_basis = zbasis.derivative_basis(1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
grad_u = d3.grad(u) + ez*lift(tau1u,-1) # First-order reduction
grad_s = d3.grad(s) + ez*lift(tau1s,-1) # First-order reduction
u_tot = u
# Problem
problem = d3.IVP([u, s, p, tau_p, tau1u, tau2u, tau1s, tau2s], namespace=locals())


# problem.add_equation("u = u_tot - U0")
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau2u,-1) = - u @ grad(u)")
# problem.add_equation("dt(u) + dot(u,grad(U0)) + dot(U0,grad(u)) + grad(p) - nu*div(grad_u) + lift(tau2u,-1) = - u @ grad(u)")
problem.add_equation("dt(s) - D*div(grad_s) + lift(tau2s, -1) = - u@grad(s)")

problem.add_equation("integ(p) = 0") # Pressure gauge
problem.add_equation("(u @ ez)(z='left') = 0")
problem.add_equation("(u @ ez)(z='right') = 0")
problem.add_equation("(div(skew(u)))(z='left') = 0")
problem.add_equation("(div(skew(u)))(z='right') = 0")

problem.add_equation("(grad_s @ ez)(z='left') = 0")
problem.add_equation("(grad_s @ ez)(z='right') = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# Background shear
u['g'][0] = 1/2 * np.tanh((z)/0.1)
# Match tracer to shear
s['g'] = 1/2 * np.tanh((z)/0.1)
# Add small vertical velocity perturbations localized to the shear layers


omega = 1.0
sigmax = 0.15
sigmaz = 0.15

u['g'][0] -= omega * -(z) * np.exp(-0.5 * ( (x - 0.5)**2 / sigmax**2 + (z)**2 / sigmaz**2) )
u['g'][1] -= omega *  (x - 0.5) * np.exp(-0.5 * ( (x - 0.5)**2 / sigmax**2 + (z)**2 / sigmaz**2) )

# Analysis
if (add_handlers):
    snapshots = solver.evaluator.add_file_handler(path + '/' + suffix + '/snapshots_target', sim_dt=0.01, max_writes=10)
    snapshots.add_task(s, name='tracer')
    snapshots.add_task(p, name='pressure')
    snapshots.add_task(-d3.div(d3.skew(u_tot)), name='vorticity')
    snapshots.add_task(d3.dot(ex, u_tot), name='ux')
    snapshots.add_task(d3.dot(ez, u_tot), name='uz')

checkpoints = solver.evaluator.add_file_handler(path + '/' + suffix + '/checkpoint_target', max_writes=2, sim_dt=stop_sim_time, mode='overwrite')
checkpoints.add_tasks(solver.state, layout='g')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((np.dot(u, ez))**2, name='w2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_w = np.sqrt(flow.max('w2'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_w))
    solver.step(timestep)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

logger.info('solve complete, sim time = {}'.format(solver.sim_time))