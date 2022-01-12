"""
Modified from: The magnetorotational instability prefers three dimensions.
3D MHD eigenvalue value problem (vector potential form)

Usage:
    mri_modes.py  [--ideal --hardwall]

Options:
    --ideal     Use Ideal MHD
    --hardwall  Use experimental boundary conditions
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

args = docopt(__doc__)
ideal = args['--ideal']
hardwall = args['--hardwall']
filename = Path("mri_options.cfg")
outbase = Path("data")

# kys = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.5, 2.75]
# kzs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.5, 2.75]
ky = 0.75
kz = 0.25
evs = []
ks = []

# Parse .cfg file to set global parameters for script
config = ConfigParser()
config.read(str(filename))

# logger.info('Running mri.py with the following parameters:')
# logger.info(config.items('parameters'))
try:
    dense = config.getboolean('solver','dense')
except:
    dense = False
if dense:
    sparse = False
    # logger.info("Using dense solver.")
    dense_threshold = config.getfloat('solver','dense_threshold')
else:
    sparse = True
    # logger.info("Using sparse solver.")

Nx = config.getint('parameters','Nx')
Lx = eval(config.get('parameters','Lx'))
# B = config.getfloat('parameters','B')

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

B = 1.0
k = 2.0
kx     =  np.pi/Lx
S      = -R*B*kx*np.sqrt(q)
f      =  R*B*kx/np.sqrt(q)
cutoff =  kx*np.sqrt(R**2 - 1)

# ftxt.write('### DIFF = ' + str(ν) + '\n')
# ftxt.write('### R = ' + str(R) + '\n')
# ftxt.write('### q = ' + str(q) + '\n')
# ftxt.write('### Bz(x) = sin' + str(k) + 'x\n')
# ftxt.write('ky, kz, GROWTH_RATE\n')

# if (ky == 0.0 and kz == 0.0):
#     continue
logger.info("Solving for mode ky = {}, kz = {}".format(ky, kz))

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
domain = de.Domain([x_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

# 3D MRI
problem_variables = ['p','vx','vy','vz','Ax','Ay','Az','Axx','Ayx','Azx', 'phi', 'ωy','ωz']
problem = de.EVP(domain, variables=problem_variables, eigenvalue='gamma')

# Local parameters
problem.parameters['S'] = S
problem.parameters['f'] = f
B = domain.new_field()
B['g'] = np.sin(2.0*x_basis.grid())
problem.parameters['B'] = B
problem.parameters['ky'] = ky
problem.parameters['kz'] = kz

# non ideal
problem.parameters['ν'] = ν
problem.parameters['η'] = η

# Operator substitutions for t derivative
problem.substitutions['dy(A)'] = "1j*ky*A"
problem.substitutions['dz(A)'] = "1j*kz*A"
problem.substitutions['Dt(A)'] = "gamma*f*A + S*x*dy(A)"
problem.substitutions['dt(A)'] = "gamma*f*A"

# non ideal
problem.substitutions['ωx'] = "dy(vz) - dz(vy)"
problem.substitutions['bx'] = "dy(Az) - dz(Ay)"
problem.substitutions['by'] = "dz(Ax) - Azx"
problem.substitutions['bz'] = "Ayx - dy(Ax)"

problem.substitutions['jx'] = "dy(bz) - dz(by)"

problem.substitutions['L(A)'] = "dy(dy(A)) + dz(dz(A))"
problem.substitutions['A_dot_grad_C(Ax, Ay, Az, C)'] = "Ax*dx(C) + Ay*dy(C) + Az*dz(C)"

problem.add_equation("dx(vx) + dy(vy) + dz(vz) = 0")

problem.add_equation("Dt(vx) -     f*vy + dx(p) - B*dz(bx) + ν*(dy(ωz) - dz(ωy)) = 0")
problem.add_equation("Dt(vy) + (f+S)*vx + dy(p) - B*dz(by) + ν*(dz(ωx) - dx(ωz)) = 0")
problem.add_equation("Dt(vz)            + dz(p) - B*dz(bz) + ν*(dx(ωy) - dy(ωx)) = 0")

problem.add_equation("ωy - dz(vx) + dx(vz) = 0")
problem.add_equation("ωz - dx(vy) + dy(vx) = 0")

# MHD equations: bx, by, bz, jxx
problem.add_equation("Axx + dy(Ay) + dz(Az) = 0")

# problem.add_equation("dt(Ax) + η * (L(Ax) + dx(Axx)) - dx(phi) = ((vy + S*x) * (bz + B) - vz*by) ")
problem.add_equation("dt(Ax) - η * (L(Ax) + dx(Axx)) - dx(phi) - (vy*B + S*x*bz) = 0")
problem.add_equation("dt(Ay) - η * (L(Ay) + dx(Ayx)) - dy(phi) + vx*B = 0")
problem.add_equation("dt(Az) - η * (L(Az) + dx(Azx)) - dz(phi) + S*x*bx = 0")

problem.add_equation("Axx - dx(Ax) = 0")
problem.add_equation("Ayx - dx(Ay) = 0")
problem.add_equation("Azx - dx(Az) = 0")

problem.add_bc("left(vx) = 0")
problem.add_bc("right(vx) = 0")
# problem.add_bc("right(p) = 0", condition="(ny == 0) and (nz == 0)")
problem.add_bc("left(ωy)   = 0")
problem.add_bc("left(ωz)   = 0")
problem.add_bc("right(ωy)  = 0")
problem.add_bc("right(ωz)  = 0")

problem.add_equation("left(Ay) = 0")
problem.add_equation("right(Ay) = 0")
problem.add_equation("left(Az) = 0")
problem.add_equation("right(Az) = 0")

# Equivalent to jxx = 0 BCs
# problem.add_equation("right(dx(dy(Ayx)) + dx(dz(Azx))) = 0")
# problem.add_equation("left(dx(dy(Ayx)) + dx(dz(Azx))) = 0")

# We get the same eigenvalues when these BCs are implemented instead of the above two
problem.add_equation("left(phi) = 0")
problem.add_equation("right(phi) = 0")

# GO
EP = Eigenproblem(problem)
t1 = time.time()
gr, idx, freq = EP.growth_rate(sparse=sparse)
t2 = time.time()
logger.info("growth rate = {}, freq = {}".format(gr,freq))
EP.solver.set_state(idx)
vx_EVP = EP.solver.state['vx']['g']
vy_EVP = EP.solver.state['vy']['g']
vz_EVP = EP.solver.state['vz']['g']
Ax_EVP = EP.solver.state['Ax']['g']
Ay_EVP = EP.solver.state['Ay']['g']
Az_EVP = EP.solver.state['Az']['g']

Nyz = 8*Nx
x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
y_basis = de.Fourier('y', Nyz, interval=(0, 8*Lx))
z_basis = de.Fourier('y', Nyz, interval=(0, 8*Lx))
domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64)
problem_variables = ['p','vx','vy','vz','Ax','Ay','Az','Axx','Ayx','Azx', 'phi', 'ωy','ωz']
problem = de.IVP(domain, variables=problem_variables)

import os
import matplotlib.pyplot as plt
path = os.path.dirname(os.path.abspath(__file__))
y, z, x = domain.all_grids()
eikykz = np.exp(1j*ky*y) * np.exp(1j*kz*z)
# print(np.shape(eikykz))
# plt.pcolor(eikykz.real[:, :, 0])
# plt.savefig(path + '/plt_domain.png')
vx = (vx_EVP * eikykz).real
vx_midy = vx[4*Nx, :, :]
vy = (vy_EVP * eikykz).real
vy_midy = vy[4*Nx, :, :]
vz = (vz_EVP * eikykz).real
vz_midy = vz[4*Nx, :, :]

z_ones = np.ones_like(z)
x_ones = np.ones_like(x)

domain_plt = [(z*x_ones)[0, :, :], (x*z_ones)[0, :, :]]
plt.subplot(3, 1, 1)
plt.pcolor(domain_plt[0], domain_plt[1], vx_midy, cmap='coolwarm')
plt.xlabel('z')
plt.ylabel('x')
plt.title('vx')
plt.subplot(3, 1, 2)
plt.pcolor(domain_plt[0], domain_plt[1], vy_midy, cmap='coolwarm')
plt.xlabel('z')
plt.ylabel('x')
plt.title('vy')
plt.subplot(3, 1, 3)
plt.pcolor(domain_plt[0], domain_plt[1], vz_midy, cmap='coolwarm')
plt.xlabel('z')
plt.ylabel('x')
plt.title('vz')
plt.suptitle('y = Ly / 2 Midplane')
plt.tight_layout()
plt.savefig(path + '/v_midy_diff2en3_R0p6.png')

vx_midz = vx[:, 4*Nx, :]
vy_midz = vy[:, 4*Nx, :]
vz_midz = vz[:, 4*Nx, :]

y_ones = np.ones_like(y)
x_ones = np.ones_like(x)

domain_plt = [(y*x_ones)[:, 0, :], (x*y_ones)[:, 0, :]]
plt.subplot(3, 1, 1)
plt.pcolor(domain_plt[0], domain_plt[1], vx_midz, cmap='coolwarm')
plt.xlabel('z')
plt.ylabel('x')
plt.title('vx')
plt.subplot(3, 1, 2)
plt.pcolor(domain_plt[0], domain_plt[1], vy_midz, cmap='coolwarm')
plt.xlabel('z')
plt.ylabel('x')
plt.title('vy')
plt.subplot(3, 1, 3)
plt.pcolor(domain_plt[0], domain_plt[1], vz_midz, cmap='coolwarm')
plt.xlabel('z')
plt.ylabel('x')
plt.title('vz')
plt.suptitle('z = Lz / 2 Midplane')
plt.tight_layout()
plt.savefig(path + '/v_midz_diff2en3_R0p6.png')


# ev_max = max(evs)
# ind_max = evs.index(ev_max)
# ftxt.write('DOMINANT MODE AT (ky,kz) = ' + str(ks[ind_max]) + '\n')
# ftxt.write('DOMINANT GROWTH RATE = ' + str(ev_max) + '\n')
# logger.info('DOMINANT MODE AT (ky,kz) = ' + str(ks[ind_max]) + '\n')
# logger.info('DOMINANT GROWTH RATE = ' + str(ev_max) + '\n')
# ftxt.write('\n')
