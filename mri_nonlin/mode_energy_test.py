from docopt import docopt
import time
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import os
import h5py
from eigentools import Eigenproblem
import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import plot_bot_2d
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

Nx = 8
Ny = 8
Nz = 8

good_ky = 2*np.pi
good_kz = 6*np.pi

Lx = 1

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
x_basis = de.Chebyshev('x', Nx, interval=(0, Lx))
y_basis = de.Fourier('y', Ny, interval=(0, Lx))
z_basis = de.Fourier('z', Nz, interval=(0, Lx))
domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)
y, z, x = [domain.grid(i) for i in range(3)]
ky, kz, kx = [domain.elements(i).squeeze() for i in range(3)]

field = domain.new_field()
field['g'] = x * np.sin(y*good_ky)*np.cos(z*good_kz)
coeffs1 = field['c'][ky == good_ky, kz == good_kz, :]
coeffs2 = field['c'][ky == good_ky, kz == -good_kz, :]
field['c'] = 0
field['c'][ky == good_ky, kz == good_kz, :]  = coeffs1
field['c'][ky == good_ky, kz == -good_kz, :] = coeffs2
#field['c'][ky == good_ky, kz == good_kz, :int(Nz/2)]  = np.random.rand(int(Nz/2)).reshape((1, int(Nz/2)))
#field['c'][ky == good_ky, kz == -good_kz, :int(Nz/2)] = np.random.rand(int(Nz/2)).reshape((1, int(Nz/2)))

#Calculation using coeffs
coeffs1 = field['c'][ky == good_ky, kz == good_kz, :]
coeffs2 = field['c'][ky == good_ky, kz == -good_kz, :]
power_coeff = np.sum(coeffs1*np.conj(coeffs1)) + np.sum(coeffs2*np.conj(coeffs2))

#Calculation in 'grid'
power_field = domain.new_field()
power_field['g'] = field['g']**2/2
power_grid = power_field.integrate()['g'].min()

print(power_grid, power_coeff, power_coeff/power_grid)
