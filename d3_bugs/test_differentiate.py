import dedalus.public as d3
import numpy as np
import matplotlib.pyplot as plt

xcoord = d3.Coordinate('x')
dist = d3.Distributor((xcoord,))
xbasis = d3.Chebyshev(xcoord, size=64, bounds=(-1, 1))

x = xbasis.local_grid(1)
f = d3.Field(dist=dist, bases=(xbasis), dtype=np.float64)
f['g'] = np.sin(x)
f.integrate(xcoord)
# fprime = f.interpolate(coord=xcoord, position=0.1)

# plt.plot(x, f['g'], label='f')
# plt.plot(x, fprime['g'], label='f\'')
# plt.legend()
# plt.show()
