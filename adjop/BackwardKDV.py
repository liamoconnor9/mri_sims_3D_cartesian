import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys

def build_problem(domain, xcoord, a, b):

    dealias = 3/2
    dtype = np.float64

    # unpack domain
    dist = domain.dist
    xbasis = domain.bases[0]
    x = dist.local_grid(xbasis)

    # Fields
    u_t = dist.Field(name='u_t', bases=xbasis)
    u = dist.Field(name='u', bases=xbasis)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Problem
    problem = d3.IVP([u_t], namespace=locals())
    problem.add_equation("dt(u_t) + a*dx(dx(u_t)) + b*dx(dx(dx(u_t)))= dx(u*u_t)")
    return problem