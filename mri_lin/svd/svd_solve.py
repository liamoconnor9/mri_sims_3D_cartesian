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
import os
import h5py
from eigentools import Eigenproblem
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

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
            div_ind = val_str.find('div')
            if (p_ind != -1):
                whole_val = val_str[:p_ind]
                decimal_val = val_str[p_ind + 1:]
                param = float(whole_val + '.' + decimal_val) * 10**(magnitude)
            elif (div_ind != -1):
                num_str = val_str[:div_ind]
                pi_ind = num_str.find('PI')
                if (pi_ind != -1):
                    num = np.pi * int(num_str[:pi_ind])
                else:
                    num = int(num_str)
                den = int(val_str[div_ind + 3:])
                param = num / den 
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

sim_suffix = 'viff1en2_eta1div7500_R1p1547_N256_noslip1_Lx1e0_ARy3PIdiv8_ARz1PIdiv1_B0PPR_U01en2'

args = docopt(__doc__)
ideal = args['--ideal']
hardwall = args['--hardwall']
append = args['--append']

sparse = True

N = int(get_param_from_suffix(sim_suffix, "N", np.NaN))
Nx_sim = N // 4
Ny = Nx_sim
Nz = N

ary = get_param_from_suffix(sim_suffix, "ARy", 8)
arz = get_param_from_suffix(sim_suffix, "ARz", 8)


Lx = 1.0
Nx = 256

Ly = Lx * ary
Lz = Lx * arz

R = 1.1547
q = 0.75

Pm = 75
nu = 1e-2
eta = nu / Pm

path = os.path.dirname(os.path.abspath(__file__))
file = h5py.File(path + '/slicepoints_s20.h5', 'r')

by = np.array(file['tasks']['by_midy'][49, 0, :, :])
u, s, vh = np.linalg.svd(by, full_matrices=False)

# [3.87527236e+01 3.89198875e-01 5.71227190e-03 8.16126736e-06, ...]

by_op = s[0] * np.matmul(u[:, 0][np.newaxis].T, vh[0, :][np.newaxis])

by_svd = s[0] * vh[0, :]

# plt.imshow(by_op - by)
# plt.colorbar()
# plt.title('error')
# plt.savefig(path + '/error.png')
# sys.exit()

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
B_vec = [i / 200 for i in range(101)]

csv_edit = 'w'
if (append):
    csv_edit = 'a'

with open('vp_growth_rates_By_svd.txt', csv_edit) as ftxt:

    ftxt.write('By, ky, kz, growth_rate, frequency\n')
    for B0 in B_vec:
        
        evs = np.zeros(nsolves)
        freqs = np.zeros(nsolves)

        kx     =  np.pi/Lx
        S      = -R*np.sqrt(q)
        f      =  R/np.sqrt(q)
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
            B.set_scales(0.25)
            B['g'] = B0*by_svd

            B_x = domain.new_field()
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

            problem.add_equation("Dt(vx) -     f*vy + dx(p) - B*dy(bx) + nu*(dy(ωz) - dz(ωy)) = 0")
            problem.add_equation("Dt(vy) + (f+S)*vx + dy(p) - B*dy(by) - bx*B_x + nu*(dz(ωx) - dx(ωz)) = 0")
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
            ftxt.write('{}, {}, {}, {}, {}\n'.format(B0, ks[ind_max][0], ks[ind_max][1], ev_max, freqs[ind_max]))
            logger.info('{}, {}, {}, {}, {}'.format(B0, ks[ind_max][0], ks[ind_max][1], ev_max, freqs[ind_max]))
        else:
            CW.Reduce(evs, evs, op=MPI.SUM, root=0)
            CW.Reduce(freqs, freqs, op=MPI.SUM, root=0)
        CW.Barrier()