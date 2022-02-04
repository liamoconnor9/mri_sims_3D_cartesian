"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. The simulation should take roughly 10
cpu-minutes to run.
The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.
To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

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
Nx, Ny, Nz = 32, 128, 128
Rayleigh = 1e6
Prandtl = 1
dealias = 3/2
stop_sim_time = 25
timestepper = d3.RK222
max_timestep = 0.01
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
A = dist.VectorField(coords, name='A', bases=(ybasis,zbasis,xbasis))

taup = dist.Field(name='taup')
# tauphi = dist.Field(name='tauphi')

tau1u = dist.VectorField(coords, name='tau1u', bases=(ybasis,zbasis))
tau2u = dist.VectorField(coords, name='tau2u', bases=(ybasis,zbasis))

tau1A = dist.VectorField(coords, name='tau1A', bases=(ybasis,zbasis))
tau2A = dist.VectorField(coords, name='tau2A', bases=(ybasis,zbasis))

# Substitutions
eta = 0.01
nu = 0.01
S = 1.0
f = 1.0

# nccs
U0 = dist.VectorField(coords, name='U0', bases=xbasis)
U0['g'][0] = S * x

fz_hat = dist.VectorField(coords, name='fz_hat', bases=xbasis)
fz_hat['g'][1] = f

# operations
b = d3.Curl(A)

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
grad_A = d3.grad(A) + ex*lift(tau1A,-1) # First-order reduction

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, phi, u, A, taup, tau1u, tau2u, tau1A, tau2A], namespace=locals())
problem.add_equation("trace(grad_u) + taup = 0")
problem.add_equation("trace(grad_A) = 0")
problem.add_equation("dt(u) + dot(u,grad(U0)) + dot(U0,grad(u)) - nu*div(grad_u) + grad(p) + lift(tau2u,-1) = cross(curl(b), b) - dot(u,grad(u)) - cross(fz_hat, u)")
problem.add_equation("dt(A) + grad(phi) - eta*div(grad_A) + lift(tau2A,-1) = cross(u, b) + cross(U0, b)")
problem.add_equation("u(x=0) = 0")
problem.add_equation("u(x=Lx) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

problem.add_equation("dot(A, ey)(x=0) = 0")
problem.add_equation("dot(A, ez)(x=0) = 0")
problem.add_equation("dot(A, ey)(x=Lx) = 0")
problem.add_equation("dot(A, ez)(x=Lx) = 0")
problem.add_equation("phi(x=0) = 0")
problem.add_equation("phi(x=Lx) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
A['g'][0] = -(np.cos(2*x) + 1) / 2.0
fh_mode = 'overwrite'

# Analysis
# snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
# snapshots.add_task(p)
# snapshots.add_task(b)
# snapshots.add_task(d3.dot(u,ey), name='uy')
# snapshots.add_task(d3.dot(u,ez), name='uz')
# snapshots.add_task(d3.dot(u,ex), name='ux')

run_suffix = 'mri_d3_test'
slicepoints = solver.evaluator.add_file_handler('slicepoints_' + run_suffix, sim_dt=0.1, max_writes=50, mode=fh_mode)

for field in [b, u]:
    for d,p in zip(('x', 'y', 'z'), (0, ar*Lx / 2.0, ar*Lx / 2.0)):
        slicepoints.add_task(d3.dot(field, ex)(x = 0), name = "{}x_mid{}".format(field, d))
        # slicepoints.add_task(d3.dot(field, ey)(x = 0), name = "{}y_mid{}".format(field, d))
        # slicepoints.add_task(d3.dot(field, ez)(x = 0), name = "{}z_mid{}".format(field, d))

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
CFL.add_velocity(b)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(d3.dot(u,u))/nu, name='Re')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()