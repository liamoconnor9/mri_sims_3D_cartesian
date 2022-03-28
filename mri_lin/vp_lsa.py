"""
Modified from: The magnetorotational instability prefers three dimensions.
3D MHD eigenvalue value problem (vector potential form)

Usage:
    vp_lsa.py  [--ideal --hardwall --append]

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

sparse = True
Nx = 256
Lx = 1.0

R = 1.1547
q = 0.75

Pm = 75
nu = 1e-2
eta = nu / Pm

# Run EVP over a grid of horizontal (y, z) wavenumbers
kys = [0.0, 0.25, 0.5, 0.75, 1.0]
kzs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]

ks = []
for ky in kys:
    for kz in kzs:
        if (ky == 0.0 and kz == 0.0):
            continue
        ks.append((ky,kz))

nsolves = len(ks)

# Run EVP for various b-field magnitudes
B_vec = [i / 100 for i in range(31)]

csv_edit = 'w'
if (append):
    csv_edit = 'a'

with open('vp_growth_rates_By_noslip_new.txt', csv_edit) as ftxt:

    ftxt.write('By, ky, kz, growth_rate, frequency\n')
    for B0 in B_vec:
        
        evs = np.zeros(nsolves)
        freqs = np.zeros(nsolves)

        kx     =  np.pi/Lx
        S      = -R*B*kx*np.sqrt(q)
        f      =  R*B*kx/np.sqrt(q)
        cutoff =  kx*np.sqrt(R**2 - 1)

        logger.info('### By = {}'.format(B0))
        for i in range(CW.rank, nsolves, CW.size):
            (ky, kz) = ks[i]

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

            # if B is an ncc we need its derivative wrt x

            B = domain.new_field()
            B_x = domain.new_field()
            B['g'] = B0*np.sin(np.pi*x_basis.grid())
            de.operators.differentiate(B, 'x', out=B_x)

            problem.parameters['B'] = B
            problem.parameters['B_x'] = B_x
            problem.parameters['ky'] = ky
            problem.parameters['kz'] = kz

            # non ideal
            problem.parameters['nu'] = nu
            problem.parameters['eta'] = eta

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

            problem.add_equation("Dt(vx) -     f*vy + dx(p) - B*dy(bx) - bx + nu*(dy(ωz) - dz(ωy)) = 0")
            problem.add_equation("Dt(vy) + (f+S)*vx + dy(p) - B*dy(by) + nu*(dz(ωx) - dx(ωz)) = 0")
            problem.add_equation("Dt(vz)            + dz(p) - B*dy(bz) + nu*(dx(ωy) - dy(ωx)) = 0")

            problem.add_equation("ωy - dz(vx) + dx(vz) = 0")
            problem.add_equation("ωz - dx(vy) + dy(vx) = 0")

            # MHD equations: bx, by, bz, jxx
            problem.add_equation("Axx + dy(Ay) + dz(Az) = 0")

            # problem.add_equation("dt(Ax) + eta * (L(Ax) + dx(Axx)) - dx(phi) = ((vy + S*x) * (bz + B) - vz*by) ")
            problem.add_equation("dt(Ax) - eta * (L(Ax) + dx(Axx)) - dx(phi) - S*x*bz + vz*B = 0")
            problem.add_equation("dt(Ay) - eta * (L(Ay) + dx(Ayx)) - dy(phi)  = 0")
            problem.add_equation("dt(Az) - eta * (L(Az) + dx(Azx)) - dz(phi) + S*x*bx - vx*B = 0")

            problem.add_equation("Axx - dx(Ax) = 0")
            problem.add_equation("Ayx - dx(Ay) = 0")
            problem.add_equation("Azx - dx(Az) = 0")

            problem.add_bc("left(vx) = 0")
            problem.add_bc("right(vx) = 0")
            # problem.add_bc("right(p) = 0", condition="(ny == 0) and (nz == 0)")

            # Stress freePE;ladL
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
            evs[i] = gr
            freqs[i] = freq

        CW.Barrier()
        if CW.rank == 0:
            CW.Reduce(MPI.IN_PLACE, evs, op=MPI.SUM, root=0)
            CW.Reduce(MPI.IN_PLACE, freqs, op=MPI.SUM, root=0)
            ev_max = max(evs)
            ind_max = np.where(evs == ev_max)[0][0]
            ftxt.write('{}, {}, {}, {}, {}\n'.format(B, ks[ind_max][0], ks[ind_max][1], ev_max, freqs[ind_max]))
            logger.info('{}, {}, {}, {}, {}'.format(B, ks[ind_max][0], ks[ind_max][1], ev_max, freqs[ind_max]))
        else:
            CW.Reduce(evs, evs, op=MPI.SUM, root=0)
            CW.Reduce(freqs, freqs, op=MPI.SUM, root=0)
        CW.Barrier()