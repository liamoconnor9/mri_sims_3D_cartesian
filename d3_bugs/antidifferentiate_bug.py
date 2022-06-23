import numpy as np
import dedalus.public as d3
from dedalus.core.domain import Domain

xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)

xbasis = d3.RealFourier(xcoord, size=8, bounds=(0, 2*np.pi), dealias=3/2)
