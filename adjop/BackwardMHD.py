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

def build_problem(domain, sim_params):

    # 3D MRI
    problem_variables = ['p_t','vx_t','vy_t','vz_t','bx_t','by_t','bz_t','ωy_t','ωz_t','jxx_t']
    problem = de.IVP(domain, variables=problem_variables, time='t')

    # Local parameters
    problem.parameters['S'] = S
    problem.parameters['f'] = f
    problem.parameters['B'] = B

    # non ideal
    problem.parameters['nu'] = nu
    problem.parameters['eta'] = eta

    # Operator substitutions for t derivative
    problem.substitutions['b_dot_grad(A)'] = "bx * dx(A) + by * dy(A) + bz * dz(A)"
    problem.substitutions['v_dot_grad(A)'] = "vx * dx(A) + vy * dy(A) + vz * dz(A)"

    # non ideal
    problem.substitutions['ωx_t'] = "dy(vz_t) - dz(vy_t)"
    problem.substitutions['jx_t'] = "dy(bz_t) - dz(by_t)"
    problem.substitutions['jy_t'] = "dz(bx_t) - dx(bz_t)"
    problem.substitutions['jz_t'] = "dx(by_t) - dy(bx_t)"
    problem.substitutions['L(A)'] = "dy(dy(A)) + dz(dz(A))"
    problem.substitutions['A_dot_grad_C(Ax, Ay, Az, C)'] = "Ax*dx(C) + Ay*dy(C) + Az*dz(C)"


    # Variable substitutions
    problem.add_equation("dx(vx_t) + dy(vy_t) + dz(vz_t) = 0")

    problem.add_equation("-dt(vx_t) = N(v_t, v) + b cross (curl b_t) + nu laplacian(u_t) + grad p_t")
    problem.add_equation("-dt(vy_t) = N(v_t, v) + b cross (curl b_t) + nu laplacian(u_t) + grad p_t")
    problem.add_equation("-dt(vz_t) = N(v_t, v) + b cross (curl b_t) + nu laplacian(u_t) + grad p_t")

    problem.add_equation("ωy_t - dz(vx_t) + dx(vz_t) = 0")
    problem.add_equation("ωz_t - dx(vy_t) + dy(vx_t) = 0")

    # MHD equations: bx, by, bz, jxx
    problem.add_equation("dx(bx) + dy(by) + dz(bz) = 0", condition='(ny != 0) or (nz != 0)')
    problem.add_equation("Dt(bx) - B*dz(vx) + eta*( dy(jz) - dz(jy) )            = b_dot_grad(vx) - v_dot_grad(bx)", condition='(ny != 0) or (nz != 0)')

    problem.add_equation("Dt(jx) - B*dz(ωx) + S*dz(bx) - eta*( dx(jxx) + L(jx) ) = b_dot_grad(ωx) - v_dot_grad(jx)"
    + "+ A_dot_grad_C(dy(bx), dy(by), dy(bz), vz) - A_dot_grad_C(dz(bx), dz(by), dz(bz), vy)"
    + "- A_dot_grad_C(dy(vx), dy(vy), dy(vz), bz) + A_dot_grad_C(dz(vx), dz(vy), dz(vz), by)", condition='(ny != 0) or (nz != 0)')

    problem.add_equation("jxx - dx(jx) = 0", condition='(ny != 0) or (nz != 0)')
    problem.add_equation("bx = 0", condition='(ny == 0) and (nz == 0)')
    problem.add_equation("by = 0", condition='(ny == 0) and (nz == 0)')
    problem.add_equation("bz = 0", condition='(ny == 0) and (nz == 0)')
    problem.add_equation("jxx = 0", condition='(ny == 0) and (nz == 0)')

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
