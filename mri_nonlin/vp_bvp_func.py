import dedalus.public as de
from mpi4py import MPI
import numpy as np
import logging as logger

CW = MPI.COMM_WORLD

Nx, Ny, Nz = 16, 16, 16
Lx, Ly, Lz = 1., 1., 1.

x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)

domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64)
problem = de.NLBVP(domain, variables=['Ax','Ay', 'Az'])

bx = domain.new_field()
by = domain.new_field()
bz = domain.new_field()

problem.parameters['bx'] = bx
problem.parameters['by'] = by
problem.parameters['bz'] = bz

problem.substitutions['curlx(Ax, Ay, Az)'] = "dy(Az) - dz(Ay)"
problem.substitutions['curly(Ax, Ay, Az)'] = "dz(Ax) - dx(Az)"
problem.substitutions['curlz(Ax, Ay, Az)'] = "dx(Ay) - dy(Ax)"

# problem.add_equation("phi = 0", condition="(ny == 0 and nz == 0)")
problem.add_equation("Ax = 0", condition="(ny == 0 and nz == 0)")
problem.add_equation("Ay = 0", condition="(ny == 0 and nz == 0)")
problem.add_equation("Az = 0", condition="(ny == 0 and nz == 0)")

# problem.add_equation("phix - dx(phi) = 0", condition="(ny != 0 or nz != 0)")

# problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0")
problem.add_equation("curlx(Ax, Ay, Az) = bx", condition="(ny != 0 or nz != 0)")
problem.add_equation("curly(Ax, Ay, Az) = by", condition="(ny != 0 or nz != 0)")
problem.add_equation("curlz(Ax, Ay, Az) = bz", condition="(ny != 0 or nz != 0)")

# problem.add_bc("left(Ay) = 0", condition="(ny != 0 or nz != 0)")
# problem.add_bc("left(Az) = 0", condition="(ny != 0 or nz != 0)")
# problem.add_bc("left(phi) = 0", condition="(ny != 0 or nz != 0)")

problem.add_bc("right(Ay) = 0", condition="(ny != 0 or nz != 0)")
problem.add_bc("right(Az) = 0", condition="(ny != 0 or nz != 0)")
# problem.add_bc("right(phi) = 0", condition="(ny != 0 or nz != 0)")

# Build solver
solver = problem.build_solver()
pert = solver.perturbations.data
pert.fill(1 + 1e-8)
while np.sum(np.abs(pert)) > 1e-8:
    solver.newton_iteration()
    logger.info(np.sum(np.abs(pert)))
# solver.solve()
