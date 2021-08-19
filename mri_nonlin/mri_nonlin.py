"""
Modified from: The magnetorotational instability prefers three dimensions.
3D MHD initial value problem (simulation)
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


##### Initial condition functions from Evan H. Anders
def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.
    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.scales
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)
    
def global_noise(domain, seed=42, **kwargs):
    """
    Create a field filled with random noise of order 1.  Modify seed to
    get varying noise, keep seed the same to directly compare runs.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]
    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    filter_field(noise_field, **kwargs)
    return noise_field


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
R      =  config.getfloat('parameters','R')
q      =  config.getfloat('parameters','q')

ν = config.getfloat('parameters','ν')
η = config.getfloat('parameters','η')

kx     =  np.pi/Lx
S      = -R*B*kx*np.sqrt(q)
f      =  0 *        R*B*kx/np.sqrt(q)
cutoff =  kx*np.sqrt(R**2 - 1)

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
y_basis = de.Fourier('y', Ny, interval=(0, Lx * 8))
z_basis = de.Fourier('z', Nz, interval=(0, Lx * 8))

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
problem.substitutions['Dt(A)'] = "dt(A) + S * x * dy(A)"
problem.substitutions['b_dot_grad(A)'] = "bx * dx(A) + by * dy(A) + bz * dz(A)"
problem.substitutions['v_dot_grad(A)'] = "vx * dx(A) + vy * dy(A) + vz * dz(A)"

# non ideal
problem.substitutions['ωx'] = "dy(vz) - dz(vy)"
problem.substitutions['jx'] = "dy(bz) - dz(by)"
problem.substitutions['jy'] = "dz(bx) - dx(bz)"
problem.substitutions['jz'] = "dx(by) - dy(bx)"
problem.substitutions['L(A)'] = "dy(dy(A)) + dz(dz(A))"

# Variable substitutions
problem.add_equation("dx(vx) + dy(vy) + dz(vz) = 0")

problem.add_equation("Dt(vx)            + dx(p) - B*dz(bx) + ν*(dy(ωz) - dz(ωy)) = b_dot_grad(bx) - v_dot_grad(vx)")
problem.add_equation("Dt(vy) + S*vx     + dy(p) - B*dz(by) + ν*(dz(ωx) - dx(ωz)) = b_dot_grad(by) - v_dot_grad(vy)")
problem.add_equation("Dt(vz)            + dz(p) - B*dz(bz) + ν*(dx(ωy) - dy(ωx)) = b_dot_grad(bz) - v_dot_grad(vz)")

problem.add_equation("ωy - dz(vx) + dx(vz) = 0")
problem.add_equation("ωz - dx(vy) + dy(vx) = 0")

# MHD equations: bx, by, bz, jxx
problem.add_equation("dx(bx) + dy(by) + dz(bz) = 0", condition='(ny != 0) or (nz != 0)')
problem.add_equation("Dt(bx) - B*dz(vx) + η*( dy(jz) - dz(jy) )            = b_dot_grad(vx) - v_dot_grad(bx)", condition='(ny != 0) or (nz != 0)')
problem.add_equation("Dt(jx) - B*dz(ωx) + S*dz(bx) - η*( dx(jxx) + L(jx) ) = b_dot_grad(ωx) - v_dot_grad(jx)", condition='(ny != 0) or (nz != 0)')
problem.add_equation("jxx - dx(jx) = 0", condition='(ny != 0) or (nz != 0)')
problem.add_equation("bx = 0", condition='(ny == 0) and (nz == 0)')
problem.add_equation("by = 0", condition='(ny == 0) and (nz == 0)')
problem.add_equation("bz = 0", condition='(ny == 0) and (nz == 0)')
problem.add_equation("jxx = 0", condition='(ny == 0) and (nz == 0)')
# if ny == 0:
# dx(bx) + dz(bz) = 0
# -Dt(dz(by)) + B*dz(dz(vy)) - eta*.... = 0
# should be ok
# similarly nz == 0 should be ok

# if ny == nz == 0:
# dx(bx) = 0
# Dt(bx) + B*dz(vx) = 0
# Dt(0) + ... = 0
# jxx = 0
# this has problems!
# magnetic variables are: bx, by, bz, jxx
# dx(bx) = 0 is the equation for bx
# jxx = 0 is the equation for jxx
# need equations for by and bz
# the equations are
# dt(by) - eta*dx(dx(by)) = -u.grad(b) + b.grad(u)

# dt(by) - eta*dx(dx(by)) = -(vx*dx(by) + vy*dy(by) + vz*dz(by)) + (bx*dx(vy) + by*dy(vy) + bz*dz(vy))
# if ny == nz == 0:
# dt(by) - eta*dx(dx(by)) = -vx*dx(by) + bx*dx(vy)
# dt(by) - eta*dx(dx(by)) = 0

# dt(bz) - eta*dx(dx(bz)) = 0
# this seems hard because it is second order in x for both by and bz.
# So instead let us just assume the initial condition hax <by>=<bz>=0, so it stays that way.

# Boundary Conditions: stress-free, perfect-conductor

problem.add_bc("left(vx) = 0")
problem.add_bc("right(vx) = 0", condition="(ny != 0) or (nz != 0)")
problem.add_bc("right(p) = 0", condition="(ny == 0) and (nz == 0)")
problem.add_bc("left(ωy)   = 0")
problem.add_bc("left(ωz)   = 0")
problem.add_bc("right(ωy)  = 0")
problem.add_bc("right(ωz)  = 0")

problem.add_bc("left(bx)   = 0", condition="(ny != 0) or (nz != 0)")
problem.add_bc("left(jxx)  = 0", condition="(ny != 0) or (nz != 0)")
problem.add_bc("right(bx) = 0", condition="(ny != 0) or (nz != 0)")
problem.add_bc("right(jxx) = 0", condition="(ny != 0) or (nz != 0)")

# setup
dt = 1e-3
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
    xb, xt = x_basis.interval
    # rand = np.random.RandomState(seed=42)
    # noise = rand.standard_normal(gshape)[slices]
    # pert =  1e-2*noise*(xt - x)*(x - xb)
    # vx['g'] = pert

    noise = global_noise(domain)
    vx['g'] += 1e0*np.cos(np.pi*(x))*noise['g']

    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)
    logger.info("loading previous state")

    # Timestepping and output
    dt = last_dt
    stop_sim_time = 20
    fh_mode = 'append'

checkpoints = solver.evaluator.add_file_handler('checkpoints_init', sim_dt=0.1, max_writes=1, mode=fh_mode)
checkpoints.add_system(solver.state)

slicepoints = solver.evaluator.add_file_handler('slicepoints_init', sim_dt=0.005, max_writes=50, mode=fh_mode)

slicepoints.add_task("interp(vx, y={})".format(Lx / 2), name="vx_midy")
slicepoints.add_task("interp(vx, z={})".format(Lx / 2), name="vx_midz")
slicepoints.add_task("interp(vx, x={})".format(0.0), name="vx_midx")

slicepoints.add_task("interp(vy, y={})".format(Lx / 2), name="vy_midy")
slicepoints.add_task("interp(vy, z={})".format(Lx / 2), name="vy_midz")
slicepoints.add_task("interp(vy, x={})".format(0.0), name="vy_midx")

slicepoints.add_task("interp(vz, y={})".format(Lx / 2), name="vz_midy")
slicepoints.add_task("interp(vz, z={})".format(Lx / 2), name="vz_midz")
slicepoints.add_task("interp(vz, x={})".format(0.0), name="vz_midx")

slicepoints.add_task("interp(bx, y={})".format(Lx / 2), name="bx_midy")
slicepoints.add_task("interp(bx, z={})".format(Lx / 2), name="bx_midz")
slicepoints.add_task("interp(bx, x={})".format(0.0), name="bx_midx")

slicepoints.add_task("interp(by, y={})".format(Lx / 2), name="by_midy")
slicepoints.add_task("interp(by, z={})".format(Lx / 2), name="by_midz")
slicepoints.add_task("interp(by, x={})".format(0.0), name="by_midx")

slicepoints.add_task("interp(bz, y={})".format(Lx / 2), name="bz_midy")
slicepoints.add_task("interp(bz, z={})".format(Lx / 2), name="bz_midz")
slicepoints.add_task("interp(bz, x={})".format(0.0), name="bz_midx")

slicepoints.add_task("integ(integ(integ(vx*vx + vy*vy + vz*vz, 'x'), 'y'), 'z')", name="ke")
slicepoints.add_task("integ(integ(integ(bx*bx + by*by + bz*bz, 'x'), 'y'), 'z')", name="be")
slicepoints.add_task("integ(integ(integ(sqrt(vx*vx + vy*vy + vz*vz), 'x'), 'y'), 'z')", name="Re")

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=dt, threshold=0.05)
CFL.add_velocities(('vx', 'vy', 'vz'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz)", name='Re')

solver.stop_sim_time = 30
solver.stop_wall_time = 1*60*60.
solver.stop_iteration = np.inf

try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
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

