import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys

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
    
# Run suffix (see mvp.sh for example)
if len(sys.argv) > 1:
    run_suffix = sys.argv[1]
    logger.info("suffix provided for write data: " + run_suffix)
else:
    logger.error("run suffix not provided")
    raise

# Parameters (Varied and Mandatory)
N = int(get_param_from_suffix(run_suffix, "N", np.NaN))
R = get_param_from_suffix(run_suffix, "R", np.NaN)
Nx = N // 4
Ny = Nz = N
diffusivities = get_param_from_suffix(run_suffix, "viff", np.NaN)

# Parameters (Varied and Optional)
Lx = get_param_from_suffix(run_suffix, "Lx", np.pi)
ar = get_param_from_suffix(run_suffix, "AR", 8)
ary = get_param_from_suffix(run_suffix, "ARy", ar)
arz = get_param_from_suffix(run_suffix, "ARz", ar)
Ly = ary * Lx
Lz = arz * Lx

q = get_param_from_suffix(run_suffix, "q", 0.75)
nu = get_param_from_suffix(run_suffix, 'nu', diffusivities)
eta = get_param_from_suffix(run_suffix, 'eta', diffusivities)
isNoSlip = get_param_from_suffix(run_suffix, 'noslip', 0)

S      = -R*np.sqrt(q)
f      =  R/np.sqrt(q)

dealias = 3/2
stop_sim_time = 10
timestepper = d3.RK222
max_timestep = 0.001
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

xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

y = dist.local_grid(ybasis)
z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# nccs
U0 = dist.VectorField(coords, name='U0', bases=xbasis)
U0['g'][0] = S * x

fz_hat = dist.VectorField(coords, name='fz_hat', bases=xbasis)
fz_hat['g'][1] = f

# Fields
p = dist.Field(name='p', bases=(ybasis,zbasis,xbasis))
phi = dist.Field(name='phi', bases=(ybasis,zbasis,xbasis))
u = dist.VectorField(coords, name='u', bases=(ybasis,zbasis,xbasis))
A = dist.VectorField(coords, name='A', bases=(ybasis,zbasis,xbasis))
b = dist.VectorField(coords, name='b', bases=(ybasis,zbasis,xbasis))

taup = dist.Field(name='taup')
# tauphi = dist.Field(name='tauphi')

tau1u = dist.VectorField(coords, name='tau1u', bases=(ybasis,zbasis))
tau2u = dist.VectorField(coords, name='tau2u', bases=(ybasis,zbasis))

tau1A = dist.VectorField(coords, name='tau1A', bases=(ybasis,zbasis))
tau2A = dist.VectorField(coords, name='tau2A', bases=(ybasis,zbasis))

# operations
# b = d3.Curl(A)
# b.store_last = True

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
problem = d3.IVP([p, phi, u, A, b, taup, tau1u, tau2u, tau1A, tau2A], namespace=locals())
problem.add_equation("trace(grad_u) + taup = 0")
problem.add_equation("trace(grad_A) = 0")
problem.add_equation("dt(u) + dot(u,grad(U0)) + dot(U0,grad(u)) - nu*div(grad_u) + grad(p) + lift(tau2u,-1) = cross(curl(b), b) - dot(u,grad(u)) - cross(fz_hat, u)")
problem.add_equation("dt(A) + grad(phi) - eta*div(grad_A) + lift(tau2A,-1) = cross(u, b) + cross(U0, b)")
problem.add_equation("b - curl(A) = 0")
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
# p.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise

rand = np.random.RandomState(seed=42)
randx = rand.rand(*np.shape(x)) / 1e10
randy = rand.rand(*np.shape(y)) / 1e10
randz = rand.rand(*np.shape(z)) / 1e10

# =741.591149
# 2022-02-10 21:45:20,595 __main__ 0/256 INFO :: Iteration=301, Time=3.516045e-02, dt=5.616887e-05, max(Re)=741.591052
u['g'][2] = x * (Lx - x)*(randx*randy*randz) / 1e10
# u['g'][1] = x * (Lx - x)

A['g'][0] = -(np.cos(2*x) + 1) / 2.0

# fh_mode = 'overwrite'
# u['g'][2] += noise / 1e1

# bz['g'] = 1e2*(np.sin((x))*np.cos(y) - 2.0/np.pi)
# A['g'][0] += -(np.cos(2*x) + 1) / 2.0
# Ay.differentiate('x', out = Ayx)
# Ax['g'] += U0 * np.cos(x) * (np.cos(z) + 1) * 2.0

# checkpoints = solver.evaluator.add_file_handler('checkpoints_' + run_suffix, sim_dt=1.0, max_writes=10, mode=fh_mode)
# checkpoints.add_system(solver.state)

slicepoints = solver.evaluator.add_file_handler('slicepoints_' + run_suffix, sim_dt=0.1, max_writes=50)

for field, field_name in [(b, 'b'), (u, 'v')]:
    for d2, unit_vec in zip(('x', 'y', 'z'), (ex, ey, ez)):
        slicepoints.add_task(d3.dot(field, unit_vec)(x = 0), name = "{}{}_mid{}".format(field_name, d2, 'x'))
        slicepoints.add_task(d3.dot(field, unit_vec)(y = 0), name = "{}{}_mid{}".format(field_name, d2, 'y'))
        slicepoints.add_task(d3.dot(field, unit_vec)(z = 0), name = "{}{}_mid{}".format(field_name, d2, 'z'))

# scalars = solver.evaluator.add_file_handler('scalars_' + run_suffix, sim_dt=0.1, max_writes=1000, mode=fh_mode)
# scalars.add_task("integ(d3.integ(d3.integ(vx*vx + vy*vy + vz*vz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke")
# scalars.add_task("integ(integ(integ(bx*bx + by*by + bz*bz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be")

# scalars.add_task("integ(integ(integ(vx*vx, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_x")
# scalars.add_task("integ(integ(integ(bx*bx, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_x")

# scalars.add_task("integ(integ(integ(vy*vy, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_y")
# scalars.add_task("integ(integ(integ((vy + S*x)*(vy + S*x), 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_y_tot")
# scalars.add_task("integ(integ(integ(by*by, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_y")

# scalars.add_task("integ(integ(integ(vz*vz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_z")
# scalars.add_task("integ(integ(integ(bz*bz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_z")
# scalars.add_task("integ(integ(integ((bz + B)*(bz + B), 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_z_tot")

# scalars.add_task("integ(integ(integ(vx*vx + (vy + S*x)*(vy + S*x) + vz*vz, 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="ke_tot")
# scalars.add_task("integ(integ(integ(bx*bx + by*by + (bz + B)*(bz + B), 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="be_tot")

# scalars.add_task("integ(integ(integ(sqrt(vx*vx + vy*vy + vz*vz), 'x'), 'y'), 'z') / (Lx**3 * ary * arz)", name="Re")



# for field in ['b', 'v']:
#     for d,p in zip(('x', 'y', 'z'), (0, ary*Lx / 2.0, arz*Lx / 2.0)):
#         slicepoints.add_task(d3.dot(field, ex)(x = 0), name = "{}x_mid{}".format(field, d))
#         # slicepoints.add_task(d3.dot(field, ey)(x = 0), name = "{}y_mid{}".format(field, d))
#         # slicepoints.add_task(d3.dot(field, ez)(x = 0), name = "{}z_mid{}".format(field, d))

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
# CFL.add_velocity(b)

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