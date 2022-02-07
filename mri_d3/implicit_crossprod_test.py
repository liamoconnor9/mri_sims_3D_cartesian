import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
logger = logging.getLogger(__name__)

# TODO: maybe fix plotting to directly handle vectors
# TODO: get unit vectors from coords?
# TODO: cleanup integ shortcuts


# Parameters
Lx = np.pi
ar = 8
Ly = Lz = ar * Lx
Nx, Ny, Nz = 4, 16, 16
stop_sim_time = 1.0
timestepper = d3.RK222
deltat = 0.01
dtype = np.float64


# Bases
coords = d3.CartesianCoordinates('y', 'z', 'x')
dist = d3.Distributor(coords, dtype=dtype)

xbasis = d3.ChebyshevT(coords['x'], size=Nz, bounds=(0, Lz))
ybasis = d3.RealFourier(coords['y'], size=Nx, bounds=(0, Lx))
zbasis = d3.RealFourier(coords['z'], size=Nx, bounds=(0, Lx))

y = dist.local_grid(ybasis)
z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# Fields
p = dist.Field(name='p', bases=(ybasis,zbasis,xbasis))
u = dist.VectorField(coords, name='u', bases=(ybasis,zbasis,xbasis))

taup = dist.Field(name='taup')
# tauphi = dist.Field(name='tauphi')

tau1u = dist.VectorField(coords, name='tau1u', bases=(ybasis,zbasis))
tau2u = dist.VectorField(coords, name='tau2u', bases=(ybasis,zbasis))

# params
nu = 0.01
S = 1.0
f = 1.0

# coriolis
fz_hat = dist.VectorField(coords, name='fz_hat', bases=xbasis)
fz_hat['g'][1] = f

# operations
ey = dist.VectorField(coords, name='ey')
ez = dist.VectorField(coords, name='ez')
ex = dist.VectorField(coords, name='ex')
ey['g'][0] = 1
ez['g'][1] = 1
ex['g'][2] = 1

integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'y'), 'z'), 'x')

lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
grad_u = d3.grad(u) + ex*lift(tau1u,-1) # First-order reduction

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, u, taup, tau1u, tau2u], namespace=locals())
problem.add_equation("trace(grad_u) + taup = 0")

# this works:
# problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) + lift(tau2u,-1) = - cross(fz_hat, u)")

# this doesn't work:
problem.add_equation("dt(u) + cross(fz_hat, u) - nu*div(grad_u) + grad(p) + lift(tau2u,-1) = 0")

problem.add_equation("u(x=0) = 0")
problem.add_equation("u(x=Lx) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
fh_mode = 'overwrite'

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(deltat)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, deltat))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()