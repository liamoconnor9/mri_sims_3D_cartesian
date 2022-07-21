from Undiffuse import Undiffuse
import dedalus.public as d3
import numpy as np
import matplotlib.pyplot as plt

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)
xbasis = d3.RealFourier(xcoord, size=128, bounds=(0, 2*np.pi), dealias=3/2)
x = dist.local_grid(xbasis)

u = dist.Field(name='u', bases = xbasis)
u['g'] = np.sin(2*x)
# ux = d3.Differentiate(u, xcoord).evaluate()
ux = Undiffuse(u, xcoord).evaluate()

u.change_scales(1)
ux.change_scales(1)

plt.plot(x, u['g'], label='u')
plt.plot(x, ux['g'], label='ux')
plt.legend()
plt.show()