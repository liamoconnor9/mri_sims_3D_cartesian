"""
The magnetorotational instability prefers three dimensions.

Single mode calculation. Returns spectrum.

Usage:
    mri_single_yz_mode.py [--ideal] <config_file> <ky> <kz>

Options:
    --ideal    Use Ideal MHD
    --hardwall Use experimental boundary conditions
"""

from docopt import docopt
import time
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import sys
import h5py
from eigentools import Eigenproblem
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

# args = docopt(__doc__)
ky = 0.0
kz = 0.5
ideal = False
hardwall = False
filename = Path("mri_options.cfg")
outbase = Path("data")

logger.info("Solving for mode ky = {}, kz = {}".format(ky, kz))
# Parse .cfg file to set global parameters for script
config = ConfigParser()
config.read(str(filename))

logger.info('Running mri.py with the following parameters:')
logger.info(config.items('parameters'))
try:
    dense = config.getboolean('solver','dense')
except:
    dense = False
if dense:
    logger.info("Using dense solver.")
    dense_threshold = config.getfloat('solver','dense_threshold')
else:
    logger.info("Using sparse solver.")

Nx = config.getint('parameters','Nx')
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
S      = -1.02
f      =  R*B*kx/np.sqrt(q)
logger.info("S = " + str(S))

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
domain = de.Domain([x_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)
B_f = domain.new_field()
x_g = x_basis.grid()
B_f['g'] = 1.
# B_f['g'] = np.sin(2*x_g)

# 3D MRI

problem_variables = ['p','vx','vy','vz','bx','by','bz']
if not ideal:
    problem_variables += ['ωy','ωz','jxx']
problem = de.EVP(domain, variables=problem_variables, eigenvalue='gamma')
problem.meta[:]['x']['dirichlet'] = True

# Local parameters

problem.parameters['S'] = S
problem.parameters['f'] = f
problem.parameters['B'] = B_f

problem.parameters['ky'] = ky
problem.parameters['kz'] = kz

if not ideal:
    problem.parameters['ν'] = ν
    problem.parameters['η'] = η

# Operator substitutions for y,z, and t derivatives

problem.substitutions['dy(A)'] = "1j*ky*A"
problem.substitutions['dz(A)'] = "1j*kz*A"
problem.substitutions['Dt(A)'] = "gamma*f*A + S*x*dy(A)"

# Variable substitutions

if not ideal:
    problem.substitutions['ωx'] = "dy(vz) - dz(vy)"
    problem.substitutions['jx'] = "dy(bz) - dz(by)"
    problem.substitutions['jy'] = "dz(bx) - dx(bz)"
    problem.substitutions['jz'] = "dx(by) - dy(bx)"
    problem.substitutions['L(A)'] = "dy(dy(A)) + dz(dz(A))"

# Hydro equations: p, vx, vy, vz, ωy, ωz

problem.add_equation("dx(vx) + dy(vy) + dz(vz) = 0")
if ideal:
    problem.add_equation("Dt(vx)  -     f*vy + dx(p) - B*dz(bx) = 0")
    problem.add_equation("Dt(vy)  + (f+S)*vx + dy(p) - B*dz(by) = 0")
    problem.add_equation("Dt(vz)             + dz(p) - B*dz(bz) = 0")

    # Frozen-in field
    problem.add_equation("Dt(bx) - B*dz(vx)        = 0")
    problem.add_equation("Dt(by) - B*dz(vy) - S*bx = 0")
    problem.add_equation("Dt(bz) - B*dz(vz)        = 0")

else:
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

problem.add_bc("left(vx)   = 0")
problem.add_bc("right(vx)  = 0")

if not ideal:
    if hardwall:
        problem.add_bc("left(vy)   = 0")
        problem.add_bc("left(vz)   = 0")
        problem.add_bc("left(jx)   = 0")
        problem.add_bc("left(dy(by)+dz(bz) + sqrt(ky**2 + kz**2)*bx) = 0")
        
        problem.add_bc("right(vy)   = 0")
        problem.add_bc("right(vz)   = 0")
        problem.add_bc("right(jx)   = 0")
        problem.add_bc("right(dy(by)+dz(bz) - sqrt(ky**2 + kz**2)*bx) = 0")
    
    else:
        problem.add_bc("left(ωy)   = 0")
        problem.add_bc("left(ωz)   = 0")
        problem.add_bc("left(bx)   = 0")
        problem.add_bc("left(jxx)  = 0")

        problem.add_bc("right(ωy)  = 0")
        problem.add_bc("right(ωz)  = 0")
        problem.add_bc("right(bx)  = 0")
        problem.add_bc("right(jxx) = 0")



# GO
EP = Eigenproblem(problem)
t1 = time.time()
gr, idx, freq = EP.growth_rate(sparse=False)
t2 = time.time()
logger.info("Solve time: {}".format(t2-t1))

gamma = EP.evalues_good

# eigenvectors = EP.solver.eigenvectors[:,EP.evalues_good_index]
index = np.argsort(-gamma.real)
logger.info("eigenvales: " + str(gamma[index]))


logger.info("Solve complete. gamma = {}".format(gamma[index[0]]))
sys.exit()
nvars = len(problem_variables)
eigvec = np.zeros((nvars,Nx),dtype=np.complex128)
EP.solver.set_state(index[0])
for k in range(nvars):
    eigvec[k,:] = EP.solver.state[problem_variables[k]]['g']

# Save either or both eigenvalues and eigenvectors to a single .h5 file
# Output file will be the .cfg file name with _output.h5
if CW.rank == 0:
    tail = '_single_mode.h5'
    if ideal:
        tail = '_ideal' + tail
    output_file_name = Path(filename.stem + tail)
    output_file = h5py.File(outbase/output_file_name, 'w')
    dset_eval = output_file.create_dataset('eigvals',data=gamma)
    dset_eval.attrs.create("ky", ky)
    dset_eval.attrs.create("kz", kz)
    dset_eval.attrs.create("R", R)
    dset_eval.attrs.create("B", B)
    dset_eval.attrs.create("q", q)
    dset_eval.attrs.create("d", Lx)
    dset_evec = output_file.create_dataset('eigvecs',data=eigvec)
    dset_evec.attrs.create('x',x_basis.grid())

    n_yz = 128
    Ly = 2*np.pi/ky
    Lz = 2*np.pi/kz
    x = x_basis.grid()
    y = np.linspace(0, Ly, n_yz, endpoint=False)
    z = np.linspace(0, Lz, n_yz, endpoint=False)

    zz,yy,xx = np.meshgrid(z,y,x,indexing='ij')
    yz_dep = np.exp(1j*(ky*yy +kz*zz))

    p = (eigvec[0,:]*yz_dep).real
    u = (eigvec[1,:]*yz_dep).real
    v = (eigvec[2,:]*yz_dep).real
    w = (eigvec[3,:]*yz_dep).real
    Bx =(eigvec[6,:]*yz_dep).real
    By =(eigvec[7,:]*yz_dep).real
    Bz =(eigvec[8,:]*yz_dep).real

    save_vtk = False
    if save_vtk:
        vtkfile = filename.stem + '.vtk'
        from pyevtk.hl import gridToVTK 
        pointData = {'p':p.copy(), 'u':u.copy(), 'v':v.copy(), 'w':w.copy(),
                     'Bx':Bx.copy(), 'By':By.copy(), 'Bz':Bz.copy()}
        print("p flags", pointData['p'].flags)

        gridToVTK(vtkfile, xx, yy, zz, pointData=pointData)

