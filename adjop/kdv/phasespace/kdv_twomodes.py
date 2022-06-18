from distutils.command.bdist import show_formats
import os
path = os.path.dirname(os.path.abspath(__file__))
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import pickle
import sys
sys.path.append(path + "/..")
import h5py
import gc
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
import ForwardKDV
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime

filename = path + '/kdv_options.cfg'
config = ConfigParser()
config.read(str(filename))

logger.info('Running diffusion_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
Lx = config.getfloat('parameters', 'Lx')
N = config.getint('parameters', 'Nx')

a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')
T = config.getfloat('parameters', 'T')
dt = config.getfloat('parameters', 'dt')
num_cp = config.getint('parameters', 'num_cp')

# Simulation Parameters
dealias = 3/2
dtype = np.float64

periodic = config.getboolean('parameters', 'periodic')
epsilon_safety = default_gamma = 0.6

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64, comm=MPI.COMM_SELF)

if periodic:
    xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, Lx), dealias=3/2)
else:
    xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(0, Lx), dealias=3/2)

domain = Domain(dist, [xbasis])
dist = domain.dist

x = dist.local_grid(xbasis)
forward_problem = ForwardKDV.build_problem(domain, xcoord, a, b)
forward_solver = forward_problem.build_solver(d3.RK443)
both = True
write_objectives = False
k1 = 1
k2 = 6
Nmodes = 20

mode1 = np.sin(k1 * 2*np.pi * x / Lx)
mode2 = np.sin(k2 * 2*np.pi * x / Lx)
mode1_f = dist.Field(name='mode1_f', bases=xbasis)
mode2_f = dist.Field(name='mode2_f', bases=xbasis)
mode1_f['g'] = mode1.copy()
mode2_f['g'] = mode2.copy()

k1_coeffs = np.linspace(0, 2, Nmodes)
k2_coeffs = np.linspace(0, 2, Nmodes)

def compute_coeffs(u, mode1_f, mode2_f):
    c1 = 2.0 / Lx * d3.Integrate(u*mode1_f).evaluate()['g'].flat[0]
    c2 = 2.0 / Lx * d3.Integrate(u*mode2_f).evaluate()['g'].flat[0]
    return (c1, c2)

def evolve(ic, forward_solver, T, dt):
    forward_solver.sim_time = 0.0
    forward_solver.stop_sim_time = T
    forward_solver.state[0].change_scales(1)
    forward_solver.state[0]['g'] = ic.copy()
    while forward_solver.proceed:
        forward_solver.step(dt)
    forward_solver.state[0].change_scales(1)
    return forward_solver.state[0]

U0 = dist.Field(name='U0', bases=xbasis)
UT = dist.Field(name='UT', bases=xbasis)
U0.change_scales(1)
UT.change_scales(1)
U0['g'] = mode1 + mode2

UT['g'] = evolve(U0['g'].copy(), forward_solver, T, dt)['g'].copy()


def evaluate_objective(u, U):
    fac2 = d3.Integrate((u - U)**2).evaluate()['g'].flat[0]
    return fac2
    # return fac1 + fac2

if (write_objectives or both):
    objectives = np.zeros((Nmodes, Nmodes))

    # ks = []
    # for k1_coeff in k1_coeffs:
    #     for k2_coeff in k2_coeffs:
    #         ks.append((k1_coeff, k2_coeff))

    inds = []
    for jrow in range(Nmodes):
        for jcol in range(Nmodes):
            inds.append((jrow, jcol))


    for indy in range(CW.rank, len(inds), CW.size):
        jrow, jcol = inds[indy]
        # k1, k2 = ks[indy]
        ic =  mode1 * k1_coeffs[jrow]
        ic += mode2 * k2_coeffs[jcol]
        u = evolve(ic, forward_solver, T, dt)
        UT['g'] = 0.0
        objectives[jrow, jcol] = evaluate_objective(u, UT)
        # print(objectives[jrow, jcol])

    CW.Barrier()
    if CW.rank == 0:
        CW.Reduce(MPI.IN_PLACE, objectives, op=MPI.SUM, root=0)
    else:
        CW.Reduce(objectives, objectives, op=MPI.SUM, root=0)

    if CW.rank == 0:
        np.savetxt(path + '/objectives2.txt', objectives)
        logger.info('saved final state')

if (not write_objectives or both):
    if CW.rank == 0:
        objectives = np.loadtxt(path + '/objectives2.txt')
        # print(objectives)
        pc = plt.pcolormesh(k1_coeffs.ravel(), k2_coeffs.ravel(), objectives.T, cmap='seismic')
        plt.colorbar(pc)
        epsilon = 1.0
        if False:
            for j in range(1):
                g1 = -0.9
                g2 = -0.9
                # g1 = 2*np.random.rand() - 1
                # g2 = 2*np.random.rand() - 1
                g = (g1 * mode1_f + g2 * mode2_f).evaluate()
                coeffs = []
                coeffs.append(compute_coeffs(g, mode1_f, mode2_f))
                for i in range(1000):
                    print(i)
                    g.change_scales(1)
                    dg = diffuse(g['g'].copy(), forward_solver, 2*T, dt)
                    g = (g - epsilon * dg).evaluate()
                    coeffs.append(compute_coeffs(g, mode1_f, mode2_f))

                cs = list(zip(*coeffs))
                c1s = cs[0]
                c2s = cs[1]
                plt.plot(c1s, c2s, color='lime', linewidth=3)
        plt.xlabel(r'$(2/L_x) \; \langle u\sin${}$x \rangle$'.format(k1))
        plt.ylabel(r'$(2/L_x) \; \langle u\sin${}$x \rangle$'.format(k2))
        plt.title(r'$\langle u^2 \rangle$')
        plt.savefig(path + '/objtest_k' + str(k1) + 'k' + str(k2) + '.png')
        plt.show()

