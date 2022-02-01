"""
Modified from: The magnetorotational instability prefers three dimensions.
3D MHD eigenvalue value problem (vector potential form)

Usage:
    vp_lsa.py  [--ideal --hardwall --append] <config_file>

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
append = args['--append']
filename = Path(args['<config_file>'])
outbase = Path("data")

kys = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.5]
# kzs = [0.25]
kzs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.5]
evs = []
ks = []

csv_edit = 'w'
if (append):
    csv_edit = 'a'

with open('vp_growth_rates_Bsin2x_noslip.txt', csv_edit) as ftxt:
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
        # dense_threshold = config.getfloat('solver','dense_threshold')
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

    ftxt.write('### DIFF = ' + str(ν) + '\n')
    ftxt.write('### R = ' + str(R) + '\n')
    ftxt.write('### q = ' + str(q) + '\n')
    ftxt.write('### Bz(x) = sin' + str(k) + 'x\n')
    ftxt.write('ky, kz, GROWTH_RATE\n')
    for ky in kys:
        for kz in kzs:
            if (ky == 0.0 and kz == 0.0):
                continue
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
            B_x = domain.new_field()
            B['g'] = np.sin(2.0*x_basis.grid())
            # 0.007634955768260147
            # de.operators.differentiate(B, 'x', out=B_x)
            B_x = B.differentiate(0)

            problem.parameters['B'] = B
            problem.parameters['B_x'] = B_x
            problem.parameters['ky'] = ky
            problem.parameters['kz'] = kz

            # non ideal
            problem.parameters['ν'] = ν
            problem.parameters['η'] = η

            # Operator substitutions for t derivative
            problem.substitutions['dy(A)'] = "1j*ky*A"
            problem.substitutions['dz(A)'] = "1j*kz*A"
            problem.substitutions['Dt(A)'] = "gamma*A + S*x*dy(A)"
            problem.substitutions['dt(A)'] = "gamma*A"

            # non ideal
            problem.substitutions['ωx'] = "dy(vz) - dz(vy)"
            problem.substitutions['bx'] = "dy(Az) - dz(Ay)"
            problem.substitutions['by'] = "dz(Ax) - Azx"
            problem.substitutions['bz'] = "Ayx - dy(Ax)"

            problem.substitutions['jx'] = "dy(bz) - dz(by)"

            problem.substitutions['L(A)'] = "dy(dy(A)) + dz(dz(A))"
            problem.substitutions['A_dot_grad_C(Ax, Ay, Az, C)'] = "Ax*dx(C) + Ay*dy(C) + Az*dz(C)"

            problem.add_equation("dx(vx) + dy(vy) + dz(vz) = 0")

            problem.add_equation("Dt(vx) -     f*vy + dx(p) - B*dz(bx) - bx*B_x + ν*(dy(ωz) - dz(ωy)) = 0")
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

            # Stress free
            # problem.add_bc("left(ωy)   = 0")
            # problem.add_bc("left(ωz)   = 0")
            # problem.add_bc("right(ωy)  = 0")
            # problem.add_bc("right(ωz)  = 0")

            # No slip
            problem.add_bc("left(vy) = 0")
            problem.add_bc("left(vz) = 0")
            problem.add_bc("right(vy) = 0")
            problem.add_bc("right(vz) = 0")

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
            ftxt.write(str(ky) + ', ' + str(kz) + ', ' + str(gr) + '\n')
            evs.append(gr)
            ks.append((ky,kz))

    ev_max = max(evs)
    ind_max = evs.index(ev_max)
    ftxt.write('DOMINANT MODE AT (ky,kz) = ' + str(ks[ind_max]) + '\n')
    ftxt.write('DOMINANT GROWTH RATE = ' + str(ev_max) + '\n')
    logger.info('DOMINANT MODE AT (ky,kz) = ' + str(ks[ind_max]) + '\n')
    logger.info('DOMINANT GROWTH RATE = ' + str(ev_max) + '\n')
    ftxt.write('\n')
