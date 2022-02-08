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
dealias = 3/2
stop_sim_time = 1.0
timestepper = d3.RK222
deltat = 0.01
dtype = np.float64

ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
else:
    logger.error("pretty sure this shouldn't happen... log2(ncpu) is not an int?")
    
logger.info("running on processor mesh={}".format(mesh))

# Bases
coords = d3.CartesianCoordinates('y', 'z', 'x')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)

xbasis = d3.ChebyshevT(coords['x'], size=Nz, bounds=(0, Lz), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nx, bounds=(0, Lx), dealias=dealias)


y = dist.local_grid(ybasis)
z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# Fields
p = dist.Field(name='p', bases=(ybasis,zbasis,xbasis))
phi = dist.Field(name='phi', bases=(ybasis,zbasis,xbasis))
u = dist.VectorField(coords, name='u', bases=(ybasis,zbasis,xbasis))
omega = dist.VectorField(coords, name='omega', bases=(ybasis,zbasis,xbasis))

taup = dist.Field(name='taup')

tau1u = dist.VectorField(coords, name='tau1u', bases=(ybasis,zbasis))
tau2u = dist.VectorField(coords, name='tau2u', bases=(ybasis,zbasis))
ex = dist.VectorField(coords, name='ex')
ex['g'][2] = 1

# Substitutions
nu = 0.01
f = 1.0

# operations
omega = d3.Curl(u)

integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'y'), 'z'), 'x')

lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
grad_u = d3.grad(u) + ex*lift(tau1u,-1) # First-order reduction

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, u, taup, tau1u, tau2u], namespace=locals())
problem.add_equation("trace(grad_u) + taup = 0")
problem.add_equation("dt(u) + grad(p) + lift(tau2u,-1) + nu*curl(omega) = 0")

# Defining omega as its own variable eliminates the error.
# problem.add_equation("omega = curl(u)")

problem.add_equation("u(x=0) = 0")
problem.add_equation("u(x=Lx) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge


# Solver
solver = problem.build_solver(timestepper)

# Initial conditions
u.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
fh_mode = 'overwrite'

solver.step(deltat)
logger.info("success")