from ast import For
from contextlib import nullcontext
import numpy as np
import os
import sys
import h5py
import gc
import dedalus.public as d3
from dedalus.core import domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
from OptimizationContext import OptimizationContext
import ForwardVB
# import BackwardMHD
# from collections import OrderedDict

# keys are forward variables
# items are (backward variables, adjoint initial condition function: i.e. ux(T) = func(ux_t(T)))
lagrangian_dict = {
    'u' : 'u_t'
}

class OptParams:
    def __init__(self, T, num_cp, dt, grad_mag):
        self.T = T
        self.num_cp = num_cp
        self.dt = dt
        self.grad_mag = grad_mag
        self.dT = T / num_cp
        self.dt_per_cp = int(self.dT // dt)

opt_params = OptParams(1.0, 1.0, 0.01, 1e-3)

# Bases
Lx = 2.
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)
xbasis = d3.ChebyshevT(xcoord, size=1024, bounds=(-Lx / 2, Lx / 2), dealias=3/2)
domain = domain.Domain(dist, [xbasis])
dist = domain.dist

x = dist.local_grid(xbasis)

forward_problem = ForwardVB.build_problem(domain, xcoord, 1e-2)
# backward_problem = BackwardMHD.build_problem(domain, sim_params)
timestepper = d3.SBDF2
write_suffix = 'vb0'

opt = OptimizationContext(domain, xcoord, forward_problem, forward_problem, lagrangian_dict, opt_params, None, timestepper, write_suffix)
n = 20
opt.ic['u']['g'] = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.2*Lx))**2) / (2*n)
opt.build_var_hotel()
opt.loop()
