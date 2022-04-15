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
    p = dist.Field(name='p', bases=bases)
    u = dist.VectorField(coords, name='u', bases=bases)
    tau_p = dist.Field(name='tau_p')
    tau1u = dist.VectorField(coords, name='tau1u', bases=(xbasis))
    tau2u = dist.VectorField(coords, name='tau2u', bases=(xbasis))

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
    grad_u = d3.grad(u) + ez*lift(tau1u,-1) # First-order reduction

    # Problem
    problem = d3.IVP([u, p, tau_p, tau1u, tau2u], namespace=locals())

    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(u) + dot(u,grad(U0)) + dot(U0,grad(u)) + grad(p) - nu*div(grad_u) + lift(tau2u,-1) = - u @ grad(u)")

    problem.add_equation("integ(p) = 0") # Pressure gauge
    problem.add_equation("u(z='left') = 0")
    problem.add_equation("u(z='right') = 0")
    return problem