import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
logger = logging.getLogger(__name__)

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

# Fields
u = dist.VectorField(coords, name='u', bases=(ybasis,zbasis,xbasis))

tau1u = dist.VectorField(coords, name='tau1u', bases=(ybasis,zbasis))
tau2u = dist.VectorField(coords, name='tau2u', bases=(ybasis,zbasis))

ex = dist.VectorField(coords, name='ex')
ex['g'][2] = 1

nu = 0.01
f = 1.0

# coriolis
fz_hat = dist.VectorField(coords, name='fz_hat', bases=xbasis)
fz_hat['g'][1] = f

lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
grad_u = d3.grad(u) + ex*lift(tau1u,-1) # First-order reduction

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([u, tau1u, tau2u], namespace=locals())

# this works:
#problem.add_equation("dt(u) - nu*div(grad_u) + lift(tau2u,-1) = - cross(fz_hat, u)")

# this doesn't work:
problem.add_equation("dt(u) - nu*div(grad_u) + cross(fz_hat, u) + lift(tau2u,-1) = 0")

problem.add_equation("u(x=0) = 0")
problem.add_equation("u(x=Lx) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.step(deltat)

