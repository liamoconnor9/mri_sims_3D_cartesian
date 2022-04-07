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
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Fields
    u_t = dist.Field(name='u_t', bases=xbasis)
    u = dist.Field(name='u', bases=xbasis)

    if (xbasis is d3.ChebyshevT):
        tau_1 = dist.Field(name='tau_1')
        tau_2 = dist.Field(name='tau_2')     
        tau_3 = dist.Field(name='tau_3')

        lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        
        # Substitutions
        u_tx = dx(u_t) + lift(tau_1) # First-order reduction
        u_txx = dx(u_tx) + lift(tau_2) # First-order reduction

        # Problem
        problem = d3.IVP([u, tau_1, tau_2, tau_3], namespace=locals())
        problem.add_equation("dt(u_t) + a*u_txx - b*dx(u_txx) + lift(tau_3) = -u*u_tx")
        # problem.add_equation("dt(u) - nu*dx(ux) + lift(tau_2) = - u*dx(u)")
        problem.add_equation("u_t(x='left') = 0")
        problem.add_equation("u_tx(x='left') = 0")
        problem.add_equation("u_t(x='right') = 0")
        return problem

    # Problem
    problem = d3.IVP([u_t], namespace=locals())
    problem.add_equation("dt(u_t) + a*dx(dx(u_t)) - b * dx(dx(dx(u_t))) = -u*dx(u_t)")
    # problem.add_equation("u_t = u")
    return problem