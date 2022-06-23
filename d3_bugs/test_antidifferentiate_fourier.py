"""Test antidifferentiate method with fourier and chebyshev bases."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers


N_range = [8, 16, 32]
L_range = [1, 2, 4]
k_range = [0, 1]
dtype_range = [np.float64, np.complex128]


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_jacobi_convert_constant(N, a, b, k, dtype, layout):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    fc = field.Field(dist=d, dtype=dtype)
    fc['g'] = 1
    fc[layout]
    f = operators.convert(fc, (b,)).evaluate()
    assert np.allclose(fc['g'], f['g'])



xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)

xbasis = d3.RealFourier(xcoord, size=8, bounds=(0, 2*np.pi), dealias=3/2)
f = dist.Field(name='f', bases=(xbasis))
f['g'] = 