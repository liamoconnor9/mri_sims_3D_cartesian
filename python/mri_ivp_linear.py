"""
Modified from: The magnetorotational instability prefers three dimensions.
linear, ideal, 3D MHD initial value problem (simulation)
"""

from docopt import docopt
import time
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import os
import h5py
from eigentools import Eigenproblem
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

ideal = True
hardwall = False
filename = Path('mri_options.cfg')
outbase = Path("data")

# Parse .cfg file to set global parameters for script
config = ConfigParser()
config.read(str(filename))

logger.info('Running mri.py with the following parameters:')
logger.info(config.items('parameters'))

Nx = config.getint('parameters','Nx')
Ny = eval(config.get('parameters','Ny'))
Nz = eval(config.get('parameters','Nz'))

Lx = eval(config.get('parameters','Lx'))

B = config.getfloat('parameters','B')

Nmodes = config.getint('parameters','Nmodes')

R      =  config.getfloat('parameters','R')
q      =  config.getfloat('parameters','q')

kymin = config.getfloat('parameters','kymin')
kymax = config.getfloat('parameters','kymax')
Nky = config.getint('parameters','Nky')

kzmin = config.getfloat('parameters','kzmin')
kzmax = config.getfloat('parameters','kzmax')
Nkz = config.getint('parameters','Nkz')

ν = config.getfloat('parameters','ν')
η = config.getfloat('parameters','η')

kx     =  np.pi/Lx
S      = -R*B*kx*np.sqrt(q)
f      =  R*B*kx/np.sqrt(q)
cutoff =  kx*np.sqrt(R**2 - 1)

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
x_basis = de.Chebyshev('r', Nx, interval=(-Lx/2, Lx/2))
y_basis = de.Fourier('y', Ny, interval=(0, Lx))
z_basis = de.Fourier('z', Nz, interval=(0, Lx))
domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)

# 3D MRI
problem_variables = ['p','vx','vy','vz','bx','by','bz']
problem = de.IVP(domain, variables=problem_variables, time='t')
# problem.meta[:]['x']['dirichlet'] = True

# Local parameters

problem.parameters['S'] = S
problem.parameters['f'] = f
problem.parameters['B'] = B

# Operator substitutions for y,z, and t derivatives
problem.substitutions['Dt(A)'] = "dt(A) + S*r*dy(A)"

# Variable substitutions
problem.add_equation("dr(vx) + dy(vy) + dz(vz) = 0")
if ideal:
    problem.add_equation("Dt(vx)  -     f*vy + dr(p) - B*dz(bx) = 0")
    problem.add_equation("Dt(vy)  + (f+S)*vx + dy(p) - B*dz(by) = 0")
    problem.add_equation("Dt(vz)             + dz(p) - B*dz(bz) = 0")

    # Frozen-in field
    problem.add_equation("Dt(bx) - B*dz(vx)        = 0")
    problem.add_equation("Dt(by) - B*dz(vy) - S*bx = 0")
    problem.add_equation("Dt(bz) - B*dz(vz)        = 0")

# Boundary Conditions: stress-free, perfect-conductor

problem.add_bc("left(vx) = 0")
# problem.add_bc("left(vy) = 0")
# problem.add_bc("left(vz) = 0")
# problem.add_bc("left(p) = 0")

problem.add_bc("right(vx) = 0", condition="(nr!=0)")
problem.add_bc("right(p) = 0", condition="(nr==0)")

# setup
dt = 0.001
solver = problem.build_solver(de.timesteppers.SBDF2)

# ICs
z = domain.grid(0)
x = domain.grid(1)
p = solver.state['p']
vx = solver.state['vx']
by = solver.state['by']
vz = solver.state['vz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-1 * noise * (zt - z) * (z - zb)
p['g'] = -(z - pert)
vx['g'] = -(z - pert) * (z - pert)
by['g'] = x * (z - pert) * (z - pert)
vz['g'] = x*x * -(z - pert) * (z - pert)

rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

solver.stop_sim_time = 100
solver.stop_wall_time = 60 * 60.
solver.stop_iteration = np.inf

try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

