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
path = os.path.dirname(os.path.abspath(__file__))
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser

filename = path + '/kdv_options.cfg'
config = ConfigParser()
config.read(str(filename))

logger.info('Running kdv_burgers.py with the following parameters:')
logger.info(config.items('parameters'))

# Parameters
Lx = config.getfloat('parameters', 'Lx')
Nx = config.getint('parameters', 'Nx')

a = config.getfloat('parameters', 'a')
b = config.getfloat('parameters', 'b')
sig = config.getfloat('parameters', 'sig_ic')
mu = config.getfloat('parameters', 'mu_ic')
ic_scale = config.getfloat('parameters', 'ic_scale')
stop_sim_time = config.getfloat('parameters', 'T')
timestep = config.getfloat('parameters', 'dt')

# Simulation Parameters
dealias = 3/2
dtype = np.float64

periodic = config.getboolean('parameters', 'periodic')
show = config.getboolean('parameters', 'show')
show_iter_cadence = config.getint('parameters', 'show_iter_cadence')


# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
u = dist.Field(name='u', bases=xbasis)

# Initial conditions
x = dist.local_grid(xbasis)
u['g'] = ic_scale*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
for k in range(1, 10):
    bfunc = dist.Field(name='bfunc', bases=xbasis)
    bfunc['g'] = np.cos(k*2*np.pi*x/Lx)
    print('k = {}, amp = {}'.format(k, d3.Integrate(bfunc * u).evaluate()['g'].flat[0]))
