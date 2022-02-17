"""
Modified from: The magnetorotational instability prefers three dimensions.
3D MHD initial value problem
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
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import plot_bot_2d
from mpi4py import MPI

CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)

def get_param_from_suffix(suffix, param_prefix, default_param):
    required = np.isnan(default_param)
    prefix_index = suffix.find(param_prefix)
    if (prefix_index == -1):
        if (required):
            logger.warning("Required parameter " + param_prefix + ": value not provided in write suffix " + suffix)
            raise 
        else:
            logger.info("Using default parameter: " + param_prefix + " = " + str(default_param))
            return default_param
    else:
        try:
            val_start_index = prefix_index + len(param_prefix)
            end_ind = suffix[val_start_index:].find("_")
            if (end_ind != -1):
                val_end_index = val_start_index + suffix[val_start_index:].find("_")
            else:
                val_end_index = val_start_index + len(suffix[val_start_index:])
            val_str = suffix[val_start_index:val_end_index]
            en_ind = val_str.find('en')
            if (en_ind != -1):
                magnitude = -int(val_str[en_ind + 2:])
                val_str = val_str[:en_ind]
            else:
                e_ind = val_str.find('e')
                if (e_ind != -1):
                    magnitude = int(val_str[e_ind + 1:])
                    val_str = val_str[:e_ind]
                else:
                    magnitude = 0.0

            p_ind = val_str.find('p')
            if (p_ind != -1):
                whole_val = val_str[:p_ind]
                decimal_val = val_str[p_ind + 1:]
                param = float(whole_val + '.' + decimal_val) * 10**(magnitude)
            else:
                param = float(val_str) * 10**(magnitude)  
            logger.info("Parameter " + param_prefix + " = " + str(param) + " : provided in write suffix")
            return param
        except Exception as e: 
            if (required):
                logger.warning("Required parameter " + param_prefix + ": failed to parse from write suffix")
                logger.info(e)
                raise 
            else:
                logger.info("Suffix parsing failed! Using default parameter: " + param_prefix + " = " + str(default_param))
                logger.info(e)
                return default_param

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
    # field.require_coeff_space()
    # field.require_grid_space()
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

if len(sys.argv) > 1:
    run_suffix = sys.argv[1]
    logger.info("suffix provided for write data: " + run_suffix)
else:
    logger.error("run suffix not provided")
    raise

# Mandatory parameters (must be provided in run suffix)
N = int(get_param_from_suffix(run_suffix, "N", np.NaN))
R = get_param_from_suffix(run_suffix, "R", np.NaN)
Nx = N // 4
Ny = Nz = N
diffusivities = get_param_from_suffix(run_suffix, "viff", np.NaN)

# Optional parameters (default values provided)
tau = get_param_from_suffix(run_suffix, "tau", 0.0)
B = get_param_from_suffix(run_suffix, "B", 0.0)
Lx = get_param_from_suffix(run_suffix, "Lx", np.pi)
q = get_param_from_suffix(run_suffix, "q", 0.75)
ar = get_param_from_suffix(run_suffix, "AR", 8)
ary = get_param_from_suffix(run_suffix, "ARy", ar)
arz = get_param_from_suffix(run_suffix, "ARz", ar)
U0 = get_param_from_suffix(run_suffix, "U0", 1)
ν = get_param_from_suffix(run_suffix, 'nu', diffusivities)
η = get_param_from_suffix(run_suffix, 'eta', diffusivities)
isNoSlip = get_param_from_suffix(run_suffix, 'noslip', 0)

S      = -R*np.sqrt(q)
f      =  R/np.sqrt(q)

# logger.info("S = " + str(S))
# logger.info("Sc = " + str(-1.0 / f))
# logger.info("S/Sc = " + str(S / -1.0 * f))

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
y_basis = de.Fourier('y', Ny, interval=(0, Lx * ary))
z_basis = de.Fourier('z', Nz, interval=(0, Lx * arz))

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
problem.parameters['ν'] = ν
problem.parameters['η'] = η

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
problem.add_equation("Dt(vx) -                f*vy + dx(p) + ν*(dy(ωz) - dz(ωy)) = b_dot_grad(bx) - v_dot_grad(vx) + B*dz(bx) + bx*B_x")
if (tau != 0.0):
    problem.add_equation("Dt(vy) +            (f+S)*vx + dy(p) + ν*(dz(ωx) - dx(ωz)) = b_dot_grad(by) - v_dot_grad(vy) + B*dz(by)", condition= "(ny != 0) or (nz != 0)")
    problem.add_equation("Dt(vy) + vy / tau + (f+S)*vx + dy(p) + ν*(dz(ωx) - dx(ωz)) = b_dot_grad(by) - v_dot_grad(vy) + B*dz(by)", condition = "(ny == 0) and (nz == 0)")
else:
    problem.add_equation("Dt(vy) +            (f+S)*vx + dy(p) + ν*(dz(ωx) - dx(ωz)) = b_dot_grad(by) - v_dot_grad(vy) + B*dz(by)")
    
problem.add_equation("Dt(vz)                       + dz(p) + ν*(dx(ωy) - dy(ωx)) = b_dot_grad(bz) - v_dot_grad(vz) + B*dz(bz)")

problem.add_equation("ωy - dz(vx) + dx(vz) = 0")
problem.add_equation("ωz - dx(vy) + dy(vx) = 0")

# MHD equations: bx, by, bz, jxx
problem.add_equation("Axx + dy(Ay) + dz(Az) = 0")

# problem.add_equation("dt(Ax) - η * (L(Ax) + dx(Axx)) - dx(phi) - (S*x*bz) = vy*B + vy*bz - vz*by ")
# problem.add_equation("dt(Ay) - η * (L(Ay) + dx(Ayx)) - dy(phi) = -vx*B + (vz*bx - vx*bz)")
# problem.add_equation("dt(Az) - η * (L(Az) + dx(Azx)) - dz(phi) + S*x*bx = (vx*by - vy*bx)")

problem.add_equation("dt(Ax) - η * (L(Ax) + dx(Axx)) - dx(phi) - (vy*B + S*x*bz) = vy*bz - vz*by ")
problem.add_equation("dt(Ay) - η * (L(Ay) + dx(Ayx)) - dy(phi) + vx*B = (vz*bx - vx*bz)")
problem.add_equation("dt(Az) - η * (L(Az) + dx(Azx)) - dz(phi) + S*x*bx = (vx*by - vy*bx)")


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

# setup
dt = 1e-4
if (diffusivities == 0.01):
    dt = 1e-2
elif (diffusivities == 0.002):
    dt = 1e-2

solver = problem.build_solver(de.timesteppers.SBDF2)
restart_state_dir = 'restart_' + run_suffix + '.h5'

if not pathlib.Path(restart_state_dir).exists():
    # ICs
    y = domain.grid(0)
    z = domain.grid(1)
    x = domain.grid(2)
    p = solver.state['p']
    vx = solver.state['vx']
    Ay = solver.state['Ay']
    Ayx = solver.state['Ayx']
    Ax = solver.state['Ax']
    Axx = solver.state['Axx']
    # bz = solver.state['bz']

    # Random perturbations, initialized globally for same results in parallel
    lshape = domain.dist.grid_layout.local_shape(scales=1)
    rand = np.random.RandomState(seed=23 + CW.rank)
    noise = rand.standard_normal(lshape)
    slices = domain.dist.grid_layout.slices(scales=1)

    # Linear background + perturbations damped at walls
    vx['g'] += noise / 1e1
    # bz['g'] = 1e2*(np.sin((x))*np.cos(y) - 2.0/np.pi)
    # Ay['g'] += -(np.cos(2*x) + 1) / 2.0
    # Ay.differentiate('x', out = Ayx)
    Ax['g'] += U0 * np.cos(x) * (np.cos(4*np.pi*z / (Lx * arz))) * 2.0
    # Ax.differentiate('y', out = Axy)
    
    filter_field(vx)
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state(restart_state_dir, -1)
    logger.info("loading previous state")

    # Timestepping and output
    dt = last_dt
    fh_mode = 'append'

NU = Scalar()
ETA = Scalar()
aspect_ratio = Scalar()
NU.value = ν
ETA.value = η
aspect_ratio = ar

checkpoints = solver.evaluator.add_file_handler('checkpoints_' + run_suffix, sim_dt=1.0, max_writes=10, mode=fh_mode)
checkpoints.add_system(solver.state)

slicepoints = solver.evaluator.add_file_handler('slicepoints_' + run_suffix, sim_dt=0.1, max_writes=50, mode=fh_mode)

slicepoints.add_task("interp(vx, y={})".format(ar * Lx / 2.0), name="vx_midy")
slicepoints.add_task("interp(vx, z={})".format(ar * Lx / 2.0), name="vx_midz")
slicepoints.add_task("interp(vx, x={})".format(0.0), name="vx_midx")

slicepoints.add_task("interp(vy, y={})".format(ar * Lx / 2.0), name="vy_midy")
slicepoints.add_task("interp(vy, z={})".format(ar * Lx / 2.0), name="vy_midz")
slicepoints.add_task("interp(vy, x={})".format(0.0), name="vy_midx")

slicepoints.add_task("interp(vy + S*x, y={})".format(ar * Lx / 2.0), name="vy_tot_midy")
slicepoints.add_task("interp(vy + S*x, z={})".format(ar * Lx / 2.0), name="vy_tot_midz")
slicepoints.add_task("interp(vy + S*x, x={})".format(0.0), name="vy_tot_midx")

slicepoints.add_task("interp(vz, y={})".format(ar * Lx / 2.0), name="vz_midy")
slicepoints.add_task("interp(vz, z={})".format(ar * Lx / 2.0), name="vz_midz")
slicepoints.add_task("interp(vz, x={})".format(0.0), name="vz_midx")

slicepoints.add_task("interp(bx, y={})".format(ar * Lx / 2.0), name="bx_midy")
slicepoints.add_task("interp(bx, z={})".format(ar * Lx / 2.0), name="bx_midz")
slicepoints.add_task("interp(bx, x={})".format(0.0), name="bx_midx")

slicepoints.add_task("interp(by, y={})".format(ar * Lx / 2.0), name="by_midy")
slicepoints.add_task("interp(by, z={})".format(ar * Lx / 2.0), name="by_midz")
slicepoints.add_task("interp(by, x={})".format(0.0), name="by_midx")

slicepoints.add_task("interp(bz, y={})".format(ar * Lx / 2.0), name="bz_midy")
slicepoints.add_task("interp(bz, z={})".format(ar * Lx / 2.0), name="bz_midz")
slicepoints.add_task("interp(bz, x={})".format(0.0), name="bz_midx")


slicepoints.add_task("integ(integ(vx, 'y'), 'z') / (Lx**2 * ary * arz)", name="vx_profs")
slicepoints.add_task("integ(integ(bx, 'y'), 'z') / (Lx**2 * ary * arz)", name="bx_profs")
slicepoints.add_task("integ(integ(vy, 'y'), 'z') / (Lx**2 * ary * arz)", name="vy_profs")
slicepoints.add_task("integ(integ(by, 'y'), 'z') / (Lx**2 * ary * arz)", name="by_profs")
slicepoints.add_task("integ(integ(vz, 'y'), 'z') / (Lx**2 * ary * arz)", name="vz_profs")
slicepoints.add_task("integ(integ(bz, 'y'), 'z') / (Lx**2 * ary * arz)", name="bz_profs")

scalars = solver.evaluator.add_file_handler('scalars_' + run_suffix, sim_dt=0.1, max_writes=1000, mode=fh_mode)
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

scalars.add_task(NU, name='nu')
scalars.add_task(ETA, name='eta')
scalars.add_task(aspect_ratio, name='ar')

path = os.path.dirname(os.path.abspath(__file__))

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.1,
                     max_change=1.5, min_change=0.5, max_dt=dt, threshold=0.05)
CFL.add_velocities(('vy', 'vz', 'vx'))

CFL_B = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.1,
                     max_change=1.5, min_change=0.5, max_dt=dt, threshold=0.05)
CFL_B.add_velocities(('by', 'bz', 'bx'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) * Lx / ν", name='Re')

solver.stop_sim_time = 1000
solver.stop_wall_time = 4*60.*60.
solver.stop_iteration = np.inf
nan_count = 0
max_nan_count = 1
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    
    while solver.ok:
        solver.step(dt)
        dt = min(CFL.compute_dt(), CFL_B.compute_dt())
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))

            if (np.isnan(flow.max('Re'))):
                nan_count += 1
                logger.warning('Max Re is nan! Strike ' + str(nan_count) + ' of ' + str(max_nan_count))
                if (nan_count >= max_nan_count):
                    sys.exit()
                    break

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

