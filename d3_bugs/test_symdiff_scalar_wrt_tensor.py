import dedalus.public as d3
import numpy as np
import matplotlib.pyplot as plt
import pytest

def test_symdiff_scalar_wrt_tensor():
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)

    xbasis = d3.RealFourier(coords['x'], size=64, bounds=(0, 2*np.pi), dealias=3/2)
    zbasis = d3.Chebyshev(coords['z'], size=64, bounds=(-1, 1), dealias=3/2)

    x, z = dist.local_grids(xbasis, zbasis)

    u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
    u['g'][0] = np.sin(x)*z
    u['g'][1] = np.cos(x)*z**2

    usqrd = d3.dot(u, u)

    assert np.allclose(usqrd.sym_diff(u).evaluate()['g'], 2*u['g'])