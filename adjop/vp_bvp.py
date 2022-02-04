
import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools


# Create bases and domain
y_basis = de.Fourier('y', 32, interval=(0, 8*np.pi))
z_basis = de.Fourier('z', 32, interval=(0, 8*np.pi))
x_basis = de.Chebyshev('x', 8, interval=(-np.pi/2.0, np.pi/2.0))
domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64)

y = domain.grid(0)
z = domain.grid(1)
x = domain.grid(2)

by = domain.new_field()
bz = domain.new_field()
bx = domain.new_field()

by['g'] = 0.
bz['g'] = np.sin(y) * np.cos(x)
bx['g'] = 0.

problem = de.LBVP(domain, variables=['Ax','Ay', 'Az', 'phi'])

problem.parameters['by'] = by
problem.parameters['bz'] = bz
problem.parameters['bx'] = bx

problem.add_equation("phi = 0", condition="(ny==0) and (nz==0)")
problem.add_equation("Ax = 0", condition="(ny==0) and (nz==0)")
problem.add_equation("Ay = 0", condition="(ny==0) and (nz==0)")
problem.add_equation("Az = 0", condition="(ny==0) and (nz==0)")

problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0", condition="(ny!=0) or (nz!=0)")
problem.add_equation("dy(Az) - dz(Ay) + dx(phi) = bx", condition="(ny!=0) or (nz!=0)")
problem.add_equation("dz(Ax) - dx(Az) + dy(phi) = by", condition="(ny!=0) or (nz!=0)")
problem.add_equation("dx(Ay) - dy(Ax) + dz(phi) = bz", condition="(ny!=0) or (nz!=0)")

problem.add_bc("left(Ay) = 0", condition="(ny!=0) or (nz!=0)")
problem.add_bc("left(Az) = 0", condition="(ny!=0) or (nz!=0)")
problem.add_bc("right(Ay) = 0", condition="(ny!=0) or (nz!=0)")
problem.add_bc("right(Az) = 0", condition="(ny!=0) or (nz!=0)")

# Build solver
solver = problem.build_solver()
solver.solve()

print('success?')

# Plot solution
Ay = solver.state['Ay']
Ax = solver.state['Ax']
Az = solver.state['Az']

bz_tilde = (Ay.differentiate('x') - Ax.differentiate('y')).evaluate()
by_tilde = (Ax.differentiate('z') - Az.differentiate('x')).evaluate()
bx_tilde = (Az.differentiate('y') - Ay.differentiate('z')).evaluate()

print(np.allclose(bx_tilde['g'], bx['g']))
print(np.allclose(by_tilde['g'], by['g']))
print(np.allclose(bz_tilde['g'], bz['g']))

# u.require_grid_space()
# plot_tools.plot_bot_2d(u)
# plt.savefig('poisson.png')
