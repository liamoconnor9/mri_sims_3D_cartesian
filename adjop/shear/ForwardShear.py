import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys
import matplotlib.pyplot as plt

def build_problem(domain, coords, Reynolds):

    # unpack domain
    dist = domain.dist
    bases = domain.bases
    x, z = dist.local_grids(bases[0], bases[1])

    # Fields
    p = dist.Field(name='p', bases=bases)
    s = dist.Field(name='s', bases=bases)
    u = dist.VectorField(coords, name='u', bases=bases)
    tau_p = dist.Field(name='tau_p')

    # Substitutions
    nu = 1 / Reynolds
    Schmidt = 1
    D = nu / Schmidt
    # x, z = dist.local_grids(xbasis, zbasis)
    # ex, ez = coords.unit_vector_fields(dist)

    # Problem
    problem = d3.IVP([u, s, p, tau_p], namespace=locals())
    problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - dot(u, grad(u))")
    problem.add_equation("dt(s) - D*lap(s) = - u@grad(s)")
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge
    return problem