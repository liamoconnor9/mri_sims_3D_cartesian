import os
path = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

# Simulation Parameters
dealias = 3/2
dtype = np.float64

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64, comm=MPI.COMM_SELF)
xbasis = d3.RealFourier(xcoord, size=16, bounds=(0, 1), dealias=3/2)
x = dist.local_grid(xbasis)

u = dist.Field(name='u', bases=xbasis)
dx = lambda A: d3.Differentiate(A, xcoord)

# Problem
problem = d3.IVP([u], namespace=locals())
problem.add_equation("dt(u) = 0")

CW.Barrier()
forward_solver = problem.build_solver(d3.RK443)
CW.Barrier()

forward_solver.state[0]['g'] = mode1 = np.sin(2*np.pi * x)
for i in range(100):
    forward_solver.step(0.001)

end_state = forward_solver.state[0]['g'].copy()
soln = np.sum(end_state * end_state)
CW.Barrier()
print('soln L2 norm = {}; rank = {}'.format(soln, CW.rank))