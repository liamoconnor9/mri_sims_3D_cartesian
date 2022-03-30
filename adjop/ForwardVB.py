"""
3D cartesian MHD initial value problem
"""

import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys

def build_problem(domain, xcoord, nu):

    dealias = 3/2
    dtype = np.float64

    # unpack domain
    dist = domain.dist
    xbasis = domain.bases[0]
    x = dist.local_grid(xbasis)

    # Fields
    u = dist.Field(name='u', bases=xbasis)
    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')

    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)
    lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    ux = dx(u) + lift(tau_1) # First-order reduction

    # Problem
    problem = d3.IVP([u, tau_1, tau_2], namespace=locals())
    problem.add_equation("dt(u) - nu*dx(ux) + lift(tau_2) = - u*dx(u)")
    problem.add_equation("u(x='left') = 0")
    problem.add_equation("u(x='right') = 0")
    
    return problem