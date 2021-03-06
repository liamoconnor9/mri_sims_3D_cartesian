"""
The magnetorotational instability prefers three dimensions.

Usage:
    mri_single_yz_mode.py  [--ideal --hardwall] <config_file>

Options:
    --ideal     Use Ideal MHD
    --hardwall  Use experimental boundary conditions
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

args = docopt(__doc__)
ideal = args['--ideal']
hardwall = args['--hardwall']
filename = Path(args['<config_file>'])
outbase = Path("data")

kys = [0.0, 0.25, 0.5]
kzs = [0.0, 0.25, 0.5]
with open('og_growth_rates.txt', 'w') as ftxt:
    ftxt.write('ky, kz, GROWTH_RATE\n')
    for ky in kys:
        for kz in kzs:
            if (ky == 0.0 and kz == 0.0):
                continue
            logger.info("Solving for mode ky = {}, kz = {}".format(ky, kz))
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

            ?? = config.getfloat('parameters','??')
            ?? = config.getfloat('parameters','??')

            kx     =  np.pi/Lx
            S      = -R*B*kx*np.sqrt(q)
            f      =  R*B*kx/np.sqrt(q)
            cutoff =  kx*np.sqrt(R**2 - 1)

            # Create bases and domain
            # Use COMM_SELF so keep calculations independent between processes
            x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
            domain = de.Domain([x_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

            # 3D MRI

            problem_variables = ['p','vx','vy','vz','bx','by','bz']
            if not ideal:
                problem_variables += ['??y','??z','jxx']
            problem = de.EVP(domain, variables=problem_variables, eigenvalue='gamma')
            problem.meta[:]['x']['dirichlet'] = True

            # Local parameters

            problem.parameters['S'] = S
            problem.parameters['f'] = f
            problem.parameters['B'] = B

            problem.parameters['ky'] = ky
            problem.parameters['kz'] = kz

            if not ideal:
                problem.parameters['??'] = ??
                problem.parameters['??'] = ??

            # Operator substitutions for y,z, and t derivatives

            problem.substitutions['dy(A)'] = "1j*ky*A"
            problem.substitutions['dz(A)'] = "1j*kz*A"
            problem.substitutions['Dt(A)'] = "gamma*f*A + S*x*dy(A)"

            # Variable substitutions

            if not ideal:
                problem.substitutions['??x'] = "dy(vz) - dz(vy)"
                problem.substitutions['jx'] = "dy(bz) - dz(by)"
                problem.substitutions['jy'] = "dz(bx) - dx(bz)"
                problem.substitutions['jz'] = "dx(by) - dy(bx)"
                problem.substitutions['L(A)'] = "dy(dy(A)) + dz(dz(A))"

            # Hydro equations: p, vx, vy, vz, ??y, ??z

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
                problem.add_equation("Dt(vx)  -     f*vy + dx(p) - B*dz(bx) + ??*(dy(??z) - dz(??y)) = 0")
                problem.add_equation("Dt(vy)  + (f+S)*vx + dy(p) - B*dz(by) + ??*(dz(??x) - dx(??z)) = 0")
                problem.add_equation("Dt(vz)             + dz(p) - B*dz(bz) + ??*(dx(??y) - dy(??x)) = 0")

                problem.add_equation("??y - dz(vx) + dx(vz) = 0")
                problem.add_equation("??z - dx(vy) + dy(vx) = 0")

                # MHD equations: bx, by, bz, jxx
                problem.add_equation("dx(bx) + dy(by) + dz(bz) = 0")
                problem.add_equation("Dt(bx) - B*dz(vx) + ??*( dy(jz) - dz(jy) )            = 0")
                problem.add_equation("Dt(jx) - B*dz(??x) + S*dz(bx) - ??*( dx(jxx) + L(jx) ) = 0")
                problem.add_equation("jxx - dx(jx) = 0")

            # Boundary Conditions: stress-free, perfect-conductor

            problem.add_bc("left(vx)   = 0")
            problem.add_bc("right(vx)  = 0")

            problem.add_bc("left(??y)   = 0")
            problem.add_bc("left(??z)   = 0")
            problem.add_bc("left(bx)   = 0")
            problem.add_bc("left(jxx)  = 0")

            problem.add_bc("right(??y)  = 0")
            problem.add_bc("right(??z)  = 0")
            problem.add_bc("right(bx)  = 0")
            problem.add_bc("right(jxx) = 0")



            # GO
            EP = Eigenproblem(problem)
            t1 = time.time()
            gr, idx, freq = EP.growth_rate(sparse=sparse)
            t2 = time.time()
            logger.info("growth rate = {}, freq = {}".format(gr,freq))
            ftxt.write(str(ky) + ', ' + str(kz) + ', ' + str(gr) + '\n')