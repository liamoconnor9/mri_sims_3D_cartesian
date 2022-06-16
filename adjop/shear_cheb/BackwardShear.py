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
    s_t = dist.Field(name='s_t', bases=bases)
    u_t = dist.VectorField(coords, name='u_t', bases=bases)

    u = dist.VectorField(coords, name='u', bases=bases)
    s = dist.Field(name='s', bases=bases)

    tau_p_t = dist.Field(name='tau_p_t')
    tau1u_t = dist.VectorField(coords, name='tau1u', bases=(xbasis))
    tau2u_t = dist.VectorField(coords, name='tau2u', bases=(xbasis))
    tau1s_t = dist.Field(name='tau1s', bases=(xbasis))
    tau2s_t = dist.Field(name='tau2s', bases=(xbasis))

    # Substitutions
    nu = 1 / Reynolds
    Schmidt = 10
    D = nu / Schmidt
    x, z = dist.local_grids(bases[0], bases[1])
    ex, ez = coords.unit_vector_fields(dist)   

    lift_basis = zbasis.derivative_basis(1) # First derivative basis
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    grad_u_t = d3.grad(u_t) + ez*lift(tau1u_t,-1) # First-order reduction
    grad_s_t = d3.grad(s_t) + ez*lift(tau1s_t,-1) # First-order reduction

    # Problem
    problem = d3.IVP([u_t, s_t, p_t, tau_p_t, tau1u_t, tau2u_t, tau1s_t, tau2s_t], namespace=locals())

    problem.add_equation("trace(grad_u_t) + tau_p_t = 0")
    problem.add_equation("dt(u_t) + grad(p_t) + nu*div(grad_u_t) + lift(tau2u_t,-1) = (dot(u_t, transpose(grad(u))) - dot(u, grad(u_t))) + s_t*grad(s)")
    problem.add_equation("dt(s_t) - D*div(grad_s_t) + lift(tau2s_t, -1) = - u@grad(s_t)")

    problem.add_equation("integ(p_t) = 0") # Pressure gauge
    problem.add_equation("(u_t @ ez)(z='left') = 0")
    problem.add_equation("(u_t @ ez)(z='right') = 0")
    problem.add_equation("(div(skew(u_t)))(z='left') = 0")
    problem.add_equation("(div(skew(u_t)))(z='right') = 0")

    problem.add_equation("(grad_s_t @ ez)(z='left') = 0")
    problem.add_equation("(grad_s_t @ ez)(z='right') = 0")

    return problem