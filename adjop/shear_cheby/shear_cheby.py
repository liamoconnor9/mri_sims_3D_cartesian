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

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shear_flow.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import os
path = os.path.dirname(os.path.abspath(__file__))
import sys

# Parameters
Lx, Lz = 4, 1
Nx, Nz = 32, 16
Reynolds = 1e3
S = 1
dealias = 3/2
stop_sim_time = 1.0
timestepper = d3.RK222
max_timestep = 1e-3
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)


# Fields
p = dist.Field(name='p', bases=(xbasis, zbasis))
s = dist.Field(name='s', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))

tau_p = dist.Field(name='tau_p')

tau1u = dist.VectorField(coords, name='tau1u', bases=(xbasis))
tau2u = dist.VectorField(coords, name='tau2u', bases=(xbasis))

# Substitutions
nu = 1 / Reynolds
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')

# nccs
U0 = dist.VectorField(coords, name='U0', bases=zbasis)
U0['g'][0] = S * z

lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
grad_u = d3.grad(u) + ez*lift(tau1u,-1) # First-order reduction

# Problem
problem = d3.IVP([u, p, tau_p, tau1u, tau2u], namespace=locals())

problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) + dot(u,grad(U0)) + dot(U0,grad(u)) + grad(p) - nu*div(grad_u) + lift(tau2u,-1) = - u @ grad(u)")

problem.add_equation("integ(p) = 0") # Pressure gauge
problem.add_equation("u(z='left') = 0")
problem.add_equation("u(z='right') = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u.fill_random('g', seed=42, distribution='normal', scale=1e-1) # Random noise


# Analysis
snapshots = solver.evaluator.add_file_handler(path + '/snapshots', sim_dt=0.1, max_writes=10)
# snapshots.add_task(s, name='tracer')
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
checkpoints = solver.evaluator.add_file_handler(path + '/checkpoint_U', max_writes=2, sim_dt=stop_sim_time, mode='overwrite')
checkpoints.add_tasks(solver.state, layout='g')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u@ez)**2, name='w2')

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

u.change_scales(1)
u_T0 = u['g'][0].copy()
u_T1 = u['g'][1].copy()
np.savetxt(path + '/shear_U0.txt', u_T0)
np.savetxt(path + '/shear_U1.txt', u_T1)
logger.info('saved final state')
