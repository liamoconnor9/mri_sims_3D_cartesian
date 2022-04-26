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
    p_t = dist.Field(name='p_t', bases=bases)
    s_t = dist.Field(name='s_t', bases=bases)
    u_t = dist.VectorField(coords, name='u_t', bases=bases)
    u = dist.VectorField(coords, name='u', bases=bases)
    tau_p_t = dist.Field(name='tau_p_t')

    # Substitutions
    nu = 1 / Reynolds
    Schmidt = 1
    D = nu / Schmidt
    # x, z = dist.local_grids(xbasis, zbasis)
    # ex, ez = coords.unit_vector_fields(dist)

    # Problem
    problem = d3.IVP([u_t, s_t, p_t, tau_p_t], namespace=locals())
    # problem.add_equation("dt(u_t) + grad(p_t) = (dot(u_t, transpose(grad(u))) - dot(u, grad(u_t)))")
    problem.add_equation("dt(u_t) + grad(p_t) + nu*lap(u_t) = (dot(u_t, transpose(grad(u))) - dot(u, grad(u_t)))")
    problem.add_equation("dt(s_t) + D*lap(s_t) = u_t@grad(s_t)")
    problem.add_equation("div(u_t) + tau_p_t = 0")
    problem.add_equation("integ(p_t) = 0") # Pressure gauge
    return problem