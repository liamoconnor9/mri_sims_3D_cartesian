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
import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import plot_bot_2d
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)

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
x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
y_basis = de.Fourier('y', Ny, interval=(0, Lx))
z_basis = de.Fourier('z', Nz, interval=(0, Lx))

ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
else:
    logger.error("pretty sure this shouldn't happen... log2(ncpu) is not an int?")
    
logger.info("running on processor mesh={}".format(mesh))

domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64, mesh=mesh)

# 3D MRI
problem_variables = ['p','vx','vy','vz','bx','by','bz','ωy','ωz','jxx']
problem = de.IVP(domain, variables=problem_variables, time='t')
# problem.meta[:]['x']['dirichlet'] = True

# Local parameters

problem.parameters['S'] = S
problem.parameters['f'] = f
problem.parameters['B'] = B

# non ideal
problem.parameters['ν'] = ν
problem.parameters['η'] = η

# Operator substitutions for t derivative
problem.substitutions['Dt(A)'] = "dt(A) + S*x*dy(A)"

# non ideal
problem.substitutions['ωx'] = "dy(vz) - dz(vy)"
problem.substitutions['jx'] = "dy(bz) - dz(by)"
problem.substitutions['jy'] = "dz(bx) - dx(bz)"
problem.substitutions['jz'] = "dx(by) - dy(bx)"
problem.substitutions['L(A)'] = "dy(dy(A)) + dz(dz(A))"

# Variable substitutions
problem.add_equation("dx(vx) + dy(vy) + dz(vz) = 0")

problem.add_equation("Dt(vx)  -     f*vy + dx(p) - B*dz(bx) + ν*(dy(ωz) - dz(ωy)) = 0")
problem.add_equation("Dt(vy)  + (f+S)*vx + dy(p) - B*dz(by) + ν*(dz(ωx) - dx(ωz)) = 0")
problem.add_equation("Dt(vz)             + dz(p) - B*dz(bz) + ν*(dx(ωy) - dy(ωx)) = 0")

problem.add_equation("ωy - dz(vx) + dx(vz) = 0")
problem.add_equation("ωz - dx(vy) + dy(vx) = 0")

# MHD equations: bx, by, bz, jxx
problem.add_equation("dx(bx) + dy(by) + dz(bz) = 0")
problem.add_equation("Dt(bx) - B*dz(vx) + η*( dy(jz) - dz(jy) )            = 0")
problem.add_equation("Dt(jx) - B*dz(ωx) + S*dz(bx) - η*( dx(jxx) + L(jx) ) = 0")
problem.add_equation("jxx - dx(jx) = 0")

# Boundary Conditions: stress-free, perfect-conductor

problem.add_equation("left(vx) = 0")
problem.add_equation("right(vx) = 0", condition="(ny!=0) or (nz!=0)")
problem.add_equation("right(bx) = 0", condition="(ny!=0) or (nz!=0)")
problem.add_equation("right(p) = 0", condition="(ny==0) and (nz==0)")

problem.add_equation("right(by) = 0", condition="(ny==0) and (nz==0)")

problem.add_bc("left(ωy)   = 0")
problem.add_bc("left(ωz)   = 0")
problem.add_bc("left(bx)   = 0")
problem.add_bc("left(jxx)  = 0")

problem.add_bc("right(ωy)  = 0")
problem.add_bc("right(ωz)  = 0")
problem.add_bc("right(jxx) = 0")

# setup
dt = 1e-6
solver = problem.build_solver(de.timesteppers.SBDF2)


if not pathlib.Path('restart.h5').exists():
    # ICs
    z = domain.grid(0)
    x = domain.grid(1)
    p = solver.state['p']
    vx = solver.state['vx']
    bx = solver.state['bx']
    vz = solver.state['vz']

    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=23)
    noise = rand.standard_normal(gshape)[slices]

    # Linear background + perturbations damped at walls
    zb, zt = z_basis.interval
    # pert =  1e-2 * noise * (zt - z) * (z - zb)
    # bx['g'] = pert

    rand = np.random.RandomState(seed=24)
    noise = rand.standard_normal(gshape)[slices]
    pert =  1e-2 * noise * (zt - z) * (z - zb)
    vx['g'] = -(z - pert)
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)
    logger.info("loading previous state")

    # Timestepping and output
    dt = last_dt
    stop_sim_time = 50
    fh_mode = 'append'

checkpoints = solver.evaluator.add_file_handler('checkpoints_mri_non', sim_dt=0.1, max_writes=6, mode=fh_mode)
checkpoints.add_system(solver.state)

slicepoints = solver.evaluator.add_file_handler('slicepoints_mri_non', sim_dt=0.01, max_writes=50, mode=fh_mode)

slicepoints.add_task("interp(vx, y={})".format(Lx / 2), name="vx_midy")
slicepoints.add_task("interp(vx, z={})".format(Lx / 2), name="vx_midz")
slicepoints.add_task("interp(vx, x={})".format(0.0), name="vx_midx")

slicepoints.add_task("interp(vy, y={})".format(Lx / 2), name="vy_midy")
slicepoints.add_task("interp(vy, z={})".format(Lx / 2), name="vy_midz")
slicepoints.add_task("interp(vy, x={})".format(0.0), name="vy_midx")

slicepoints.add_task("interp(vz, y={})".format(Lx / 2), name="vz_midy")
slicepoints.add_task("interp(vz, z={})".format(Lx / 2), name="vz_midz")
slicepoints.add_task("interp(vz, x={})".format(0.0), name="vz_midx")
slicepoints.add_task("integ(integ(integ(sqrt(vx*vx + vy*vy + vz*vz), 'x'), 'y'), 'z')", name="Re")

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('vx', 'vy', 'vz'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz)", name='Re')

solver.stop_sim_time = 30
solver.stop_wall_time = 8 * 60 * 60.
solver.stop_iteration = np.inf

try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

