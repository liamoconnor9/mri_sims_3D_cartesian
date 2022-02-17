import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
logger = logging.getLogger(__name__)

# TODO: maybe fix plotting to directly handle vectors
# TODO: get unit vectors from coords?
# TODO: cleanup integ shortcuts


# Parameters
Lx = np.pi
ar = 8
Ly = Lz = ar * Lx
Nx, Ny, Nz = 32, 128, 128
timestepper = d3.RK222
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('y', 'z', 'x')
dist = d3.Distributor(coords, dtype=dtype)

xbasis = d3.ChebyshevT(coords['x'], size=Nz, bounds=(0, Lz))
ybasis = d3.RealFourier(coords['y'], size=Nx, bounds=(0, Lx))
zbasis = d3.RealFourier(coords['z'], size=Nx, bounds=(0, Lx))

y = dist.local_grid(ybasis)
z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# Fields
A = dist.VectorField(coords, name='A', bases=(ybasis,zbasis,xbasis))

# operations
b = d3.Curl(A)
