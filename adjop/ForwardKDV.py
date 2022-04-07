import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys
import matplotlib.pyplot as plt

def build_problem(domain, xcoord, a, b):

    # unpack domain
    dist = domain.dist
    xbasis = domain.bases[0]

    # Fields
    u = dist.Field(name='u', bases=xbasis)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Problem
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) - a*dx(dx(u)) - b*dx(dx(dx(u))) = - u*dx(u)")
    return problem