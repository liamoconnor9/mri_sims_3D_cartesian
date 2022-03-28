"""
3D cartesian MRI initial value problem using vector potential formulation
Modified from: The magnetorotational instability prefers three dimensions.
Usage:
    mri_vp.py <config_file> <dir> <run_suffix>
"""

from unicodedata import decimal
from docopt import docopt
import time
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import os
import sys
import h5py
import gc
import pickle
import dedalus.public as de
from dedalus.core.field import Scalar
from dedalus.core import pencil
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import plot_bot_2d
from mpi4py import MPI
from natsort import natsorted

CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)

##### Initial condition functions from Evan H. Anders

def vp_bvp_func(domain, by, bz, bx):
    problem = de.LBVP(domain, variables=['Ax','Ay', 'Az', 'phi'])

    problem.parameters['by'] = by
    problem.parameters['bz'] = bz
    problem.parameters['bx'] = bx

    problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0")
    problem.add_equation("dy(Az) - dz(Ay) + dx(phi) = bx")
    problem.add_equation("dz(Ax) - dx(Az) + dy(phi) = by")
    problem.add_equation("dx(Ay) - dy(Ax) + dz(phi) = bz")

    problem.add_bc("left(Ay) = 0", condition="(ny!=0) or (nz!=0)")
    problem.add_bc("left(Az) = 0", condition="(ny!=0) or (nz!=0)")
    problem.add_bc("right(Ay) = 0", condition="(ny!=0) or (nz!=0)")
    problem.add_bc("right(Az) = 0", condition="(ny!=0) or (nz!=0)")

    problem.add_bc("left(Ax) = 0", condition="(ny==0) and (nz==0)")
    problem.add_bc("left(Ay) = 0", condition="(ny==0) and (nz==0)")
    problem.add_bc("left(Az) = 0", condition="(ny==0) and (nz==0)")
    problem.add_bc("left(phi) = 0", condition="(ny==0) and (nz==0)")

    # Build solver
    solver = problem.build_solver()
    solver.solve()

    # Plot solution
    Ay = solver.state['Ay']
    Az = solver.state['Az']
    Ax = solver.state['Ax']
    phi = solver.state['phi']

    return Ay['g'], Az['g'], Ax['g']

Pm_vec = np.linspace(35, 25, 100)

args = docopt(__doc__)
filename = Path(args['<config_file>'])
script_dir = args['<dir>']
run_suffix = args['<run_suffix>']

config = ConfigParser()
config.read(str(filename))

logger.info('Running mri_vp.py with the following parameters:')
logger.info(config.items('parameters'))

restart = config.getboolean('parameters','restart')

Ny = config.getint('parameters','Ny')
Ly = eval(config.get('parameters','Ly'))

Nz = config.getint('parameters','Nz')
Lz = eval(config.get('parameters','Lz'))

Nx = config.getint('parameters','Nx')
Lx = eval(config.get('parameters','Lx'))

B = config.getfloat('parameters','B')

R      =  config.getfloat('parameters','R')
q      =  config.getfloat('parameters','q')

nu = config.getfloat('parameters','nu')
Pm = config.getfloat('parameters','Pm')
eta = nu / Pm

S      = -R*np.sqrt(q)
f      =  R/np.sqrt(q)

tau = config.getfloat('parameters','tau')
isNoSlip = config.getboolean('parameters','isNoSlip')

ary = Ly / Lx
arz = Lz / Lx

# Evolution params
dt = config.getfloat('parameters', 'dt')
stop_sim_time = config.getfloat('parameters', 'stop_sim_time')
wall_time = 60. * 60. * config.getfloat('parameters', 'wall_time_hr')

# Create bases and domain
x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)

ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
else:
    logger.error("pretty sure this shouldn't happen... log2(ncpu) is not an int?")
    
logger.info("running on processor mesh={}".format(mesh))
CW.barrier()

domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64, mesh=mesh)

# 3D MRI
problem_variables = ['p', 'vx','vy','vz','Ax','Ay','Az','Axx','Ayx','Azx', 'phi', 'ωy','ωz']
problem = de.IVP(domain, variables=problem_variables, time='t')

# Local parameters
problem.parameters['S'] = S
problem.parameters['Lx'] = Lx
problem.parameters['ary'] = ary
problem.parameters['arz'] = arz
problem.parameters['f'] = f

# B = domain.new_field()
# B_x = domain.new_field()
# B['g'] = np.sin(2.0*domain.grid(2))
# de.operators.differentiate(B, 'x', out=B_x)
# B_x = B.differentiate(2)

problem.parameters['B'] = B
problem.parameters['B_x'] = 0
problem.parameters['tau'] = tau

# non ideal
problem.parameters['nu'] = nu
problem.parameters['eta'] = eta

# Operator substitutions for t derivative
problem.substitutions['Dt(A)'] = "dt(A) + S * x * dy(A)"
problem.substitutions['b_dot_grad(A)'] = "bx * dx(A) + by * dy(A) + bz * dz(A)"
problem.substitutions['v_dot_grad(A)'] = "vx * dx(A) + vy * dy(A) + vz * dz(A)"

# non ideal
problem.substitutions['ωx'] = "dy(vz) - dz(vy)"
problem.substitutions['bx'] = "dy(Az) - dz(Ay)"
problem.substitutions['by'] = "dz(Ax) - Azx"
problem.substitutions['bz'] = "Ayx - dy(Ax)"

problem.substitutions['jx'] = "dy(bz) - dz(by)"

problem.substitutions['L(A)'] = "dy(dy(A)) + dz(dz(A))"
problem.substitutions['A_dot_grad_C(Ax, Ay, Az, C)'] = "Ax*dx(C) + Ay*dy(C) + Az*dz(C)"

problem.add_equation("dx(vx) + dy(vy) + dz(vz) = 0")

# tau dampening timescale
problem.add_equation("Dt(vx)                  + dx(p) + nu*(dy(ωz) - dz(ωy)) = b_dot_grad(bx) - v_dot_grad(vx) + f*vy")
if (tau != 0.0):
    problem.add_equation("Dt(vy) +            (S)*vx + dy(p) + nu*(dz(ωx) - dx(ωz)) = b_dot_grad(by) - v_dot_grad(vy) - f*vx", condition= "(ny != 0) or (nz != 0)")
    problem.add_equation("Dt(vy) + vy / tau + (S)*vx + dy(p) + nu*(dz(ωx) - dx(ωz)) = b_dot_grad(by) - v_dot_grad(vy) - f*vx", condition = "(ny == 0) and (nz == 0)")
else:
    problem.add_equation("Dt(vy) +            (S)*vx + dy(p) + nu*(dz(ωx) - dx(ωz)) = b_dot_grad(by) - v_dot_grad(vy) - f*vx")
    
problem.add_equation("Dt(vz)                       + dz(p) + nu*(dx(ωy) - dy(ωx)) = b_dot_grad(bz) - v_dot_grad(vz)")

problem.add_equation("ωy - dz(vx) + dx(vz) = 0")
problem.add_equation("ωz - dx(vy) + dy(vx) = 0")

# MHD equations: bx, by, bz, jxx
problem.add_equation("Axx + dy(Ay) + dz(Az) = 0")

# problem.add_equation("dt(Ax) - eta * (L(Ax) + dx(Axx)) - dx(phi) - (S*x*bz) = vy*B + vy*bz - vz*by ")
# problem.add_equation("dt(Ay) - eta * (L(Ay) + dx(Ayx)) - dy(phi) = -vx*B + (vz*bx - vx*bz)")
# problem.add_equation("dt(Az) - eta * (L(Az) + dx(Azx)) - dz(phi) + S*x*bx = (vx*by - vy*bx)")

problem.add_equation("dt(Ax) - eta * (L(Ax) + dx(Axx)) - dx(phi) - (vy*B) = S*x*bz + vy*bz - vz*by ")
problem.add_equation("dt(Ay) - eta * (L(Ay) + dx(Ayx)) - dy(phi) + vx*B = (vz*bx - vx*bz)")
problem.add_equation("dt(Az) - eta * (L(Az) + dx(Azx)) - dz(phi) = - S*x*bx + (vx*by - vy*bx)")


problem.add_equation("Axx - dx(Ax) = 0")
problem.add_equation("Ayx - dx(Ay) = 0")
problem.add_equation("Azx - dx(Az) = 0")
# problem.add_equation("bz - Ayx + dy(Ax) = 0")

problem.add_bc("left(vx) = 0")
problem.add_bc("right(vx) = 0", condition="(ny != 0) or (nz != 0)")
problem.add_bc("right(p) = 0", condition="(ny == 0) and (nz == 0)")

if (isNoSlip):
    logger.info('Adding no slip BCs')
    problem.add_bc("left(vy) = 0")
    problem.add_bc("left(vz) = 0")
    problem.add_bc("right(vy) = 0")
    problem.add_bc("right(vz) = 0")
else:
    logger.info('Adding stress free BCs')
    problem.add_bc("left(ωy)   = 0")
    problem.add_bc("left(ωz)   = 0")
    problem.add_bc("right(ωy)  = 0")
    problem.add_bc("right(ωz)  = 0")

problem.add_equation("left(Ay) = 0")
problem.add_equation("right(Ay) = 0")
problem.add_equation("left(Az) = 0")
problem.add_equation("right(Az) = 0")
problem.add_equation("left(phi) = 0")
problem.add_equation("right(phi) = 0")

solver = problem.build_solver(de.timesteppers.SBDF2)

solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = wall_time
solver.stop_iteration = np.inf

if not restart:
    # ICs
    y = domain.grid(0)
    z = domain.grid(1)
    x = domain.grid(2)
    p = solver.state['p']
    vx = solver.state['vx']
    vy = solver.state['vy']
    vz = solver.state['vz']

    phi = solver.state['phi']
    Ay = solver.state['Ay']
    Ayx = solver.state['Ayx']
    Ax = solver.state['Ax']
    Axx = solver.state['Axx']
    Az = solver.state['Az']
    Azx = solver.state['Azx']
    # bz = solver.state['bz']

    # Random perturbations, initialized globally for same results in parallel
    lshape = domain.dist.grid_layout.local_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)

    file = h5py.File('/home3/loconno2/mri/mri_nonlin/AC_Pm35/checkpoints/checkpoints_s10.h5', 'r')
    cp_index = -1
   
    Ay['g'] = file['tasks/Ay'][cp_index, :, :, :][slices]
    Az['g'] = file['tasks/Az'][cp_index, :, :, :][slices]
    Ax['g'] = file['tasks/Ax'][cp_index, :, :, :][slices]

    vy['g'] = file['tasks/vy'][cp_index, :, :, :][slices]
    vz['g'] = file['tasks/vz'][cp_index, :, :, :][slices]
    vx['g'] = file['tasks/vx'][cp_index, :, :, :][slices]

    p['g'] = file['tasks/p'][cp_index, :, :, :][slices]
    phi['g'] = file['tasks/phi'][cp_index, :, :, :][slices]

    Ay.differentiate('x', out = Ayx)
    Az.differentiate('x', out = Azx)
    Ax.differentiate('x', out = Axx)
    fh_mode = 'overwrite'

else:
    # Restart
    restart_dir = script_dir + run_suffix + '/checkpoints'
    checkpoint_names = [name for name in os.listdir(restart_dir) if name[-3:] == '.h5']
    last_checkpoint = natsorted(checkpoint_names)[-1]
    last_checkpoint_path = restart_dir + '/' + last_checkpoint
    write, last_dt = solver.load_state(last_checkpoint_path, -1)
    logger.info("loading previous state")

    # Timestepping and output
    dt = last_dt
    fh_mode = 'append'

checkpoints = solver.evaluator.add_file_handler('{}/checkpoints'.format(run_suffix), sim_dt=20.0, max_writes=10, mode=fh_mode)
checkpoints.add_system(solver.state)

slicepoints = solver.evaluator.add_file_handler('{}/slicepoints'.format(run_suffix), sim_dt=2.0, max_writes=50, mode=fh_mode)

slicepoints.add_task("interp(vx, y={})".format(ary * Lx / 2.0), name="vx_midy")
slicepoints.add_task("interp(vx, z={})".format(arz * Lx / 2.0), name="vx_midz")
slicepoints.add_task("interp(vx, x={})".format(0.0), name="vx_midx")

slicepoints.add_task("interp(vy, y={})".format(ary * Lx / 2.0), name="vy_midy")
slicepoints.add_task("interp(vy, z={})".format(arz * Lx / 2.0), name="vy_midz")
slicepoints.add_task("interp(vy, x={})".format(0.0), name="vy_midx")

slicepoints.add_task("interp(vy + S*x, y={})".format(ary * Lx / 2.0), name="vy_tot_midy")
slicepoints.add_task("interp(vy + S*x, z={})".format(arz * Lx / 2.0), name="vy_tot_midz")
slicepoints.add_task("interp(vy + S*x, x={})".format(0.0), name="vy_tot_midx")

slicepoints.add_task("interp(vz, y={})".format(ary * Lx / 2.0), name="vz_midy")
slicepoints.add_task("interp(vz, z={})".format(arz * Lx / 2.0), name="vz_midz")
slicepoints.add_task("interp(vz, x={})".format(0.0), name="vz_midx")

slicepoints.add_task("interp(bx, y={})".format(ary * Lx / 2.0), name="bx_midy")
slicepoints.add_task("interp(bx, z={})".format(arz * Lx / 2.0), name="bx_midz")
slicepoints.add_task("interp(bx, x={})".format(0.0), name="bx_midx")

slicepoints.add_task("interp(by, y={})".format(ary * Lx / 2.0), name="by_midy")
slicepoints.add_task("interp(by, z={})".format(arz * Lx / 2.0), name="by_midz")
slicepoints.add_task("interp(by, x={})".format(0.0), name="by_midx")

slicepoints.add_task("interp(bz, y={})".format(ary * Lx / 2.0), name="bz_midy")
slicepoints.add_task("interp(bz, z={})".format(arz * Lx / 2.0), name="bz_midz")
slicepoints.add_task("interp(bz, x={})".format(0.0), name="bz_midx")


slicepoints.add_task("integ(integ(vx, 'y'), 'z') / (Lx**2 * ary * arz)", name="vx_profs")
slicepoints.add_task("integ(integ(bx, 'y'), 'z') / (Lx**2 * ary * arz)", name="bx_profs")
slicepoints.add_task("integ(integ(vy, 'y'), 'z') / (Lx**2 * ary * arz)", name="vy_profs")
slicepoints.add_task("integ(integ(by, 'y'), 'z') / (Lx**2 * ary * arz)", name="by_profs")
slicepoints.add_task("integ(integ(vz, 'y'), 'z') / (Lx**2 * ary * arz)", name="vz_profs")
slicepoints.add_task("integ(integ(bz, 'y'), 'z') / (Lx**2 * ary * arz)", name="bz_profs")

scalars = solver.evaluator.add_file_handler('{}/scalars'.format(run_suffix), sim_dt=0.1, max_writes=1000, mode=fh_mode)
scalars.add_task("integ(integ(integ(vx*vx + vy*vy + vz*vz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke")
scalars.add_task("integ(integ(integ(bx*bx + by*by + bz*bz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be")

scalars.add_task("integ(integ(integ(vx*vx, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_x")
scalars.add_task("integ(integ(integ(bx*bx, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_x")

scalars.add_task("integ(integ(integ(vy*vy, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_y")
scalars.add_task("integ(integ(integ((vy + S*x)*(vy + S*x), 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_y_tot")
scalars.add_task("integ(integ(integ(by*by, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_y")

scalars.add_task("integ(integ(integ(vz*vz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_z")
scalars.add_task("integ(integ(integ(bz*bz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_z")
scalars.add_task("integ(integ(integ((bz + B)*(bz + B), 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_z_tot")

scalars.add_task("integ(integ(integ(vx*vx + (vy + S*x)*(vy + S*x) + vz*vz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_tot")
scalars.add_task("integ(integ(integ(bx*bx + by*by + (bz + B)*(bz + B), 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_tot")

scalars.add_task("integ(integ(integ(sqrt(vx*vx + vy*vy + vz*vz), 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="Re")

path = os.path.dirname(os.path.abspath(__file__))

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=dt, threshold=0.05)
CFL.add_velocities(('vy', 'vz', 'vx'))
CFL.add_velocities(('by', 'bz', 'bx'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) / nu", name='Re')
flow.add_property("sqrt(bx*bx + by*by + bz*bz)", name='BE')

nan_count = 0
max_nan_count = 1
update_cadence = 2000
stop = solver.stop_sim_time
num_iter = int(stop // dt)
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    
    for iter in range(num_iter):
        if (iter % update_cadence == 0):
            solver.problem.namespace['eta'].value = nu / Pm_vec[iter // update_cadence]
            logger.info('Updating Pm to: {}'.format(Pm_vec[iter // update_cadence]))
            pencil.build_matrices(solver.pencils, problem, ['M', 'L'])
            
            # solver = problem.build_solver(de.timesteppers.SBDF2)
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            string = 'Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt)
            string += ', Max Re = %e' %flow.max('Re')
            string += ', Max Rm = %e' %flow.max('BE')
            logger.info(string)

            if (np.isnan(flow.max('Re')) or np.isnan(flow.max('BE'))):
                nan_count += 1
                logger.warning('Max Re is nan! Strike ' + str(nan_count) + ' of ' + str(max_nan_count))
                if (nan_count >= max_nan_count):
                    sys.exit()
                    break

except:
    # logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

