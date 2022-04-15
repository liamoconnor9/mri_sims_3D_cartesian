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
    xbasis, zbasis = bases[0], bases[1]

    # Fields
    p_t = dist.Field(name='p_t', bases=bases)
    u_t = dist.VectorField(coords, name='u_t', bases=bases)
    u = dist.VectorField(coords, name='u', bases=bases)
    tau_p_t = dist.Field(name='tau_p_t')
    tau1u_t = dist.VectorField(coords, name='tau1u_t', bases=(xbasis))
    tau2u_t = dist.VectorField(coords, name='tau2u_t', bases=(xbasis))

    # Substitutions
    nu = 1 / Reynolds
    S = 1
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')

    # nccs
    U0 = dist.VectorField(coords, name='U0', bases=zbasis)
    U0['g'][0] = S * z

    lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    grad_u_t = d3.grad(u_t) + ez*lift(tau1u_t,-1) # First-order reduction

    # Problem
    problem = d3.IVP([u_t, p_t, tau_p_t, tau1u_t, tau2u_t], namespace=locals())
    problem.add_equation("dt(u_t) + grad(p_t) + nu*div(grad_u_t) + lift(tau2u_t, -1) = u_t @ transpose(grad(u + U0)) - (u + U0) @ grad(u_t)")
    problem.add_equation("trace(grad_u_t) + tau_p_t = 0")
    problem.add_equation("integ(p_t) = 0") # Pressure gauge
    problem.add_equation("u_t(z='left') = 0")
    problem.add_equation("u_t(z='right') = 0")
    return problem