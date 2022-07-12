import os
path = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pickle
import sys
sys.path.append(path + "/..")
sys.path.append(path + "/../diffusion")
sys.path.append(path + "/../kdv")
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import ForwardKDV
import ForwardDiffusion
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
import ast
import publication_settings
import matplotlib
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
plt.rcParams.update({'figure.figsize': [3.4, 3.4]})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

filename = path + '/twomodes_options.cfg'
config = ConfigParser()
config.read(str(filename))

logger.info('Running diffusion_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

problem = str(config.get('parameters', 'problem'))
both = config.getboolean('parameters', 'both')
write_objectives = config.getboolean('parameters', 'write_objectives')


ks = ast.literal_eval(config.get('parameters', 'ks'))
k1, k2 = ks[0], ks[1]

target_coeffs = ast.literal_eval(config.get('parameters', 'target_coeffs'))
k1_range = ast.literal_eval(config.get('parameters', 'k1_range'))
k2_range = ast.literal_eval(config.get('parameters', 'k2_range'))
R = config.getfloat('parameters', 'R')
Nmodes = config.getint('parameters', 'Nmodes')

# Parameters
Lx = eval(config.get('parameters', 'Lx'))
N = config.getint('parameters', 'Nx')

a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')
T = config.getfloat('parameters', 'T')

dt = config.getfloat('parameters', 'dt')
num_cp = config.getint('parameters', 'num_cp')

a_str = str(a).replace('.', 'p')
b_str = str(b).replace('.', 'p')
T_str = str(T).replace('.', 'p')
kt1_str = str(target_coeffs[0]).replace('.', 'p')
kt2_str = str(target_coeffs[1]).replace('.', 'p')
R_str = str(R).replace('.', 'p')

objectives_str = path + '/objectives_a' + a_str + 'b' + b_str + 'T' + T_str + 'R' + R_str + 'kt1' + kt1_str + 'kt2' + kt2_str + '.txt'
save_dir = path + '/objtest_a' + a_str + 'b' + b_str + 'T' + T_str + 'R' + R_str + 'kt1' + kt1_str + 'kt2' + kt2_str + '.png'

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
CW.Barrier()

if problem == 'kdv':
    forward_problem = ForwardKDV.build_problem(domain, xcoord, a, b)

elif problem == 'diffusion':
    forward_problem = ForwardDiffusion.build_problem(domain, xcoord, a, b)

else:
    logger.error('problem not recognized')
    raise

# match problem:
#     case 'kdv':
#         forward_problem = ForwardKDV.build_problem(domain, xcoord, a, b)

#     case 'diffusion':
#         forward_problem = ForwardDiffusion.build_problem(domain, xcoord, a, b)

#     case _:
#         logger.error('problem not recognized')
#         raise

CW.Barrier()
forward_solver = forward_problem.build_solver(d3.RK443)
CW.Barrier()

mode1 = np.sin(k1 * 2*np.pi * x / Lx)
mode2 = np.sin(k2 * 2*np.pi * x / Lx)
mode1_f = dist.Field(name='mode1_f', bases=xbasis)
mode2_f = dist.Field(name='mode2_f', bases=xbasis)
mode1_f['g'] = mode1.copy()
mode2_f['g'] = mode2.copy()

k1_coeffs = np.linspace(target_coeffs[0] - R, target_coeffs[0] + R, Nmodes)
k2_coeffs = np.linspace(target_coeffs[1] - R, target_coeffs[1] + R, Nmodes)

# k1_coeffs = np.linspace(k1_range[0], k1_range[1], Nmodes)
# k2_coeffs = np.linspace(k2_range[0], k2_range[1], Nmodes)

def compute_coeffs(u, mode1_f, mode2_f):
    c1 = 2.0 / Lx * d3.Integrate(u*mode1_f).evaluate()['g'].flat[0]
    c2 = 2.0 / Lx * d3.Integrate(u*mode2_f).evaluate()['g'].flat[0]
    return (c1, c2)

def evolve(ic, fsolve, T, dt):
    fsolve.sim_time = 0.0
    fsolve.stop_sim_time = T
    fsolve.state[0].change_scales(1)
    fsolve.state[0]['g'] = ic.copy()

    while fsolve.proceed:
        fsolve.step(dt)
    fsolve.state[0].change_scales(1)
    # CW.barrier()
    return fsolve.state[0]['g'].copy()

u2 = dist.Field(name='u2', bases=xbasis)
U0 = dist.Field(name='U0', bases=xbasis)
UT = dist.Field(name='UT', bases=xbasis)
U0.change_scales(1)
UT.change_scales(1)
U0['g'] = target_coeffs[0]*mode1 + target_coeffs[1]*mode2

utg = UT['g'].copy() * 0.0

if (write_objectives or both):

    logger.info('running target simulation')
    if (CW.rank == 0):
        utg = evolve(U0['g'].copy(), forward_solver, T, dt)
        CW.Reduce(MPI.IN_PLACE, utg, op=MPI.SUM, root=0)
    else:
        CW.Reduce(utg, utg, op=MPI.SUM, root=0)
    # logger.info(UT['g'])

    CW.Bcast([utg, MPI.DOUBLE], root=0)
    UT['g'] = utg.copy()

    CW.Barrier()
    CW.barrier()
    # print(np.sum(UT['g']**2))

    logger.info('target state computed')
    # sys.exit()

    def evaluate_objective(u, U):
        # fac2 = np.sum((u['g'])**2)
        fac2 = d3.Integrate((u - U)**2).evaluate()['g'].flat[0] / Lx
        # print('obj = {}, rank = {}'.format(fac2, CW.rank))
        return fac2
        # return fac1 + fac2

    objectives = np.zeros((Nmodes, Nmodes))

    inds = []
    for jrow in range(Nmodes):
        for jcol in range(Nmodes):
            inds.append((jrow, jcol))
    CW.Barrier()

    for indy in range(CW.rank, len(inds), CW.size):
        jrow, jcol = inds[indy]
        mode1 = np.sin(k1 * 2*np.pi * x / Lx)
        mode2 = np.sin(k2 * 2*np.pi * x / Lx)
        # ic =  mode1 * k1_coeffs[jrow]
        # ic += mode2 * k2_coeffs[jcol]
        # ic = mode1 + mode2
        ic = k1_coeffs[jrow] * mode1 + k2_coeffs[jcol] * mode2
        u2.change_scales(1)
        UT.change_scales(1)
        u2['g'] = evolve(ic, forward_solver, T, dt)
        objectives[jrow, jcol] = evaluate_objective(u2, UT)
        # sys.exit()
        # print(objectives[jrow, jcol])

    CW.Barrier()
    if CW.rank == 0:
        CW.Reduce(MPI.IN_PLACE, objectives, op=MPI.SUM, root=0)
    else:
        CW.Reduce(objectives, objectives, op=MPI.SUM, root=0)

    if CW.rank == 0:
        np.savetxt(objectives_str, objectives)
        logger.info('saved final state')


if (not write_objectives or both):
    if CW.rank == 0:
        def fmt(x):
            print(str(x))
            return '  ' + str(x) + '  '

        objectives = np.loadtxt(objectives_str)
        # print(objectives)
        pc = plt.pcolormesh(k1_coeffs.ravel(), k2_coeffs.ravel(), objectives.T, cmap='GnBu')
        cb = plt.colorbar(pc, fraction=0.046, pad=0.15)
        plt.scatter([target_coeffs[0]], [target_coeffs[1]], s=60, marker='*', color='k', label = 'target')
        cont = plt.contour(k1_coeffs.ravel(), k2_coeffs.ravel(), objectives.T, levels=[0.00001, 0.00002, 0.00004], colors=['k', 'k'])
        plt.clabel(cont, cont.levels, inline=True, fmt=fmt, fontsize=7)
        plt.legend(loc='lower right')
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
        plt.xlabel(r'$(2/L_x) \; \langle u(x, 0) \, \sin${}$x \rangle$'.format(k1))
        plt.ylabel(r'$(2/L_x) \; \langle u(x, 0) \, \sin${}$x \rangle$'.format(k2))
        plt.title(r'$\langle (u(T) - U(T))^2 \rangle$; a = {}, b = {}, T = {}'.format(a, b, T), pad=20)
        import types
        def bottom_offset(self, bboxes, bboxes2):
            bottom = self.axes.bbox.ymin
            self.offsetText.set(va="top", ha="left")
            self.offsetText.set_position(
                    (0, bottom - self.OFFSETTEXTPAD * self.figure.dpi / 72.0))
        cb.formatter.set_scientific(True)
        cb.formatter.set_powerlimits((0,0))

        def register_bottom_offset(axis, func):
            axis._update_offset_text_position = types.MethodType(func, axis)
        register_bottom_offset(cb.ax.yaxis, bottom_offset)
        plt.setp(plt.axes().get_xticklabels(), rotation=20, horizontalalignment='right')

        cb.update_ticks()
        plt.axes().set_aspect('equal')
        plt.savefig(save_dir)
        logger.info('image saved to file: ' + save_dir)
        plt.show()
