"""
Dedalus script simulating the 1D Korteweg-de Vries / Burgers equation.
This script demonstrates solving a 1D initial value problem and produces
a space-time plot of the solution. It should take just a few seconds to
run (serial only).
We use a Fourier basis to solve the IVP:
    dt(u) + u*dx(u) = a*dx(dx(u)) + b*dx(dx(dx(u)))
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import os

# Simulation Parameters
Lx = 10
Nx = 256
a = 0.01
b = 0.0
dealias = 3/2
dtype = np.float64
stop_sim_time = 3

timestepper = d3.SBDF2
epsilon_safety = 1
timestep = 5e-4

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

# Fields
u = dist.Field(name='u', bases=xbasis)

# Substitutions
dx = lambda A: d3.Differentiate(A, xcoord)

# Problem
problem = d3.IVP([u], namespace=locals())
problem.add_equation("dt(u) - a*dx(dx(u)) - b*dx(dx(dx(u))) = - u*dx(u)")

# Initial conditions
x = dist.local_grid(xbasis)
mu = 5.5
sig = 0.5
u['g'] = 1*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Main loop
# Main loop
u.change_scales(1)
fig = plt.figure()
p, = plt.plot(x, u['g'])
fig.canvas.draw()
title = plt.title('t=%f' %solver.sim_time)

for iter in range(int(solver.stop_sim_time // timestep) + 1):
    solver.step(timestep)
    # if solver.iteration % 100 == 0:
    logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    if solver.iteration % 20 == 0:
        u.change_scales(1)
        p.set_ydata(u['g'])
        plt.pause(1e-10)
        fig.canvas.draw()


logger.info('solve complete, sim time = {}'.format(solver.sim_time))
u.change_scales(1)
u_T = u['g'].copy()
path = os.path.dirname(os.path.abspath(__file__))
np.savetxt(path + '/kdv_U.txt', u_T)
logger.info('saved final state')

# # Plot
# plt.figure(figsize=(6, 4))
# plt.pcolormesh(x.ravel(), np.array(t_list), np.array(u_list), cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
# plt.xlim(0, Lx)
# plt.ylim(0, stop_sim_time)
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title(f'KdV-Burgers, (a,b)=({a},{b})')
# plt.tight_layout()
# plt.savefig('kdv_burgers.pdf')
# plt.savefig('kdv_burgers.png', dpi=200)