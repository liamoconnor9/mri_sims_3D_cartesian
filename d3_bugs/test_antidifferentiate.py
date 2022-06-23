"""Test antidifferentiate method with fourier and chebyshev bases."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers


N_range = [8, 16, 32]
L_range = [1, 2, 4]
k_range = [1, 2, 3]
# dtype_range = [np.float64, np.complex128]


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('L', L_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_antidifferentiate_chebyshev(N, L, k, dtype, layout):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Chebyshev(c, size=N, bounds=(0, L))
    x = b.local_grid(1)
    fc = field.Field(dist=d, dtype=dtype)
    fs = field.Field(dist=d, dtype=dtype)
    fc['g'] = np.cos(k*x)
    fc.antidifferentiate(b, )
    fc[layout]
    f = operators.convert(fc, (b,)).evaluate()
    assert np.allclose(fc['g'], f['g'])



xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)

xbasis = d3.RealFourier(xcoord, size=8, bounds=(0, 2*np.pi), dealias=3/2)
f = dist.Field(name='f', bases=(xbasis))
f['g'] = 