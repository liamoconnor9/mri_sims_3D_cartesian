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
    xcoord, zcoord = dist.coords[0], dist.coords[1]

    # Fields
    psi = dist.Field(name='psi', bases=(xbasis, zbasis))
    p = dist.Field(name='p', bases=(xbasis, zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
    s = dist.Field(name='s', bases=(xbasis, zbasis))

    tau_p = dist.Field(name='tau_p')
    tau1u = dist.VectorField(coords, name='tau1u', bases=(xbasis))
    tau2u = dist.VectorField(coords, name='tau2u', bases=(xbasis))
    tau1s = dist.Field(name='tau1s', bases=(xbasis))
    tau2s = dist.Field(name='tau2s', bases=(xbasis))
    tau1psi = dist.Field(name='tau1psi', bases=(xbasis))
    tau2psi = dist.Field(name='tau2psi', bases=(xbasis))

    # Substitutions
    nu = 1 / Reynolds
    Schmidt = 1
    D = nu / Schmidt
    S = 0
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')

    lift_basis = zbasis.derivative_basis(1) # First derivative basis
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    grad_u = d3.grad(u) + ez*lift(tau1u,-1) # First-order reduction
    grad_s = d3.grad(s) + ez*lift(tau1s,-1) # First-order reduction
    grad_psi = d3.grad(psi) + ez*lift(tau1psi,-1) # First-order reduction

    dx = lambda A: d3.Differentiate(A, xcoord)
    dz = lambda A: d3.Differentiate(A, zcoord)
    omega = dx(u @ ez) - dz(u @ ex)
    upsi = d3.skew(grad_psi)
    disc = (u - upsi) @ (u - upsi)

    # Problem
    problem = d3.IVP([u, s, p, psi, tau_p, tau1psi, tau2psi, tau1u, tau2u, tau1s, tau2s], namespace=locals())
    # problem = d3.IVP([u, s, p, tau_p, tau1u, tau2u, tau1s, tau2s], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau2u,-1) = - u @ grad(u)")
    problem.add_equation("dt(s) - D*div(grad_s) + lift(tau2s, -1) = - u@grad(s)")
    problem.add_equation("div(grad_psi) - omega + lift(tau2psi, -1) = 0")

    problem.add_equation("integ(p) = 0") # Pressure gauge
    problem.add_equation("psi(z='left') = 0")
    problem.add_equation("psi(z='right') = 0")

    problem.add_equation("(u @ ez)(z='left') = 0")
    problem.add_equation("(u @ ez)(z='right') = 0")
    problem.add_equation("(div(skew(u)))(z='left') = 0")
    problem.add_equation("(div(skew(u)))(z='right') = 0")
    problem.add_equation("(grad_s @ ez)(z='left') = 0")
    problem.add_equation("(grad_s @ ez)(z='right') = 0")

    return problem