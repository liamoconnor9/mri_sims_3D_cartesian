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
    
    u = dist.Field(name='u', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)

    if (not isinstance(xbasis, d3.RealFourier)):
        tau_1 = dist.Field(name='tau_1')
        tau_2 = dist.Field(name='tau_2')     
        tau_3 = dist.Field(name='tau_3')

        lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        
        # Substitutions
        ux = dx(u) + lift(tau_1) # First-order reduction
        uxx = dx(ux) + lift(tau_2) # First-order reduction

        # Problem
        problem = d3.IVP([u, tau_1, tau_2, tau_3], namespace=locals())
        problem.add_equation("dt(u) - a*dx(ux) - b*dx(uxx) + lift(tau_3) = -u*ux")

        problem.add_equation("u(x='left') = 0")
        problem.add_equation("u(x='right') = 0")
        problem.add_equation("ux(x='left') = 0")

        # problem.add_equation("u(x='left') - u(x='right') = 0")
        # problem.add_equation("ux(x='left') - ux(x='right') = 0")
        # problem.add_equation("uxx(x='left') - uxx(x='right') = 0")
        return problem
        

    # Problem
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) - a*dx(dx(u)) - b*dx(dx(dx(u))) = -u*dx(u)")
    return problem