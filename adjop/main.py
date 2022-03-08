from ast import For
from contextlib import nullcontext
import numpy as np
import os
import sys
import h5py
import gc
import dedalus.public as d3
from dedalus.core import domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
from OptimizationContext import OptimizationContext
import ForwardMHD
# import BackwardMHD
from collections import OrderedDict

def build_domain(Ly, Lz, Lx, Ny, Nz, Nx, dealias, dtype):
    coords = d3.CartesianCoordinates('y', 'z', 'x')
    ncpu = MPI.COMM_WORLD.size
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    else:
        logger.error("pretty sure this shouldn't happen... log2(ncpu) is not an int?")
    logger.info("running on processor mesh={}".format(mesh))
    dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
    # print(coords)
    # coords = dist.coords
    # print(coords)
    ybasis = d3.RealFourier(coords[0], size=Ny, bounds=(0, Ly), dealias=dealias)
    zbasis = d3.RealFourier(coords[1], size=Nz, bounds=(0, Lz), dealias=dealias)
    xbasis = d3.ChebyshevT(coords[2], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0), dealias=dealias)
    return (domain.Domain(dist, [ybasis, zbasis, xbasis]), coords)



# TODO: replace below with .cfg file
q = 0.75
R = 0.6
sim_params = {
    'isNoSlip' : 1,
    'S' : -R*np.sqrt(q),
    'f' : R/np.sqrt(q),
    'nu' : 0.01,
    'eta' : 0.01
    }

# keys are forward variables
# items are (backward variables, adjoint initial condition function: i.e. ux(T) = func(ux_t(T)))
lagrangian_dict = {
    'u' : ('u_t', lambda p: -p),
    'b' : ('b_t', lambda p: -p)
}

class OptParams:
    def __init__(self, T, num_cp, dt, grad_mag):
        self.T = T
        self.num_cp = num_cp
        self.dt = dt
        self.grad_mag = grad_mag
        self.dT = T / num_cp
        self.dt_per_cp = self.dT // dt

opt_params = OptParams(1.0, 1.0, 0.01, 1e-3)
domain, coords = build_domain(8*np.pi, 8*np.pi, np.pi, 8, 8, 4, 3/2, np.float64)

forward_problem = ForwardMHD.build_problem(domain, coords, sim_params)
# backward_problem = BackwardMHD.build_problem(domain, sim_params)
timestepper = d3.SBDF2
write_suffix = 'init0'

opt = OptimizationContext(domain, coords, forward_problem, forward_problem, lagrangian_dict, opt_params, sim_params, timestepper, write_suffix)
opt.ic.field_dict['u']['g'] = np.sin(domain.grid(0))
opt.build_var_hotel()
opt.loop()



sys.exit()
class Optimization_params:
    # T, Ncheckpoints, 
    # dt ## should be factor of T / N_checkpoints?
    # run_suffix = 'init'
    # M0 = 'initial magnetic energy budget'
    # U0 = 'initial kinetic energy budget'
    def __init__(self, T, Ncheckpoints, Ndt, M0, U0):
        self.T = T
        self.Ncheckpoints = Ncheckpoints
        self.Ndt = Ndt
        self.M0 = M0
        self.U0 = U0
    
class OptimizationContext:
    def __init__(self, write_suffix, opt_params, opt_tol, sim_params):
        self.write_suffix = write_suffix
        self.opt_params = opt_params
        self.sim_params = sim_params
        self.opt_tol = opt_tol
        self.domain = None
        self.opt_loop = None
        self.ic = None # list of fields (field system)

    def build_domain(self, Lx, ar, Ny, Nz, Nx):
        x_basis = de.Chebyshev('x', Nx, interval=(-Lx/2, Lx/2))
        y_basis = de.Fourier('y', Ny, interval=(0, Lx * ar))
        z_basis = de.Fourier('z', Nz, interval=(0, Lx * ar))    
        ncpu = MPI.COMM_WORLD.size
        log2 = np.log2(ncpu)
        if log2 == int(log2):
            mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
        else:
            logger.error("pretty sure this shouldn't happen... log2(ncpu) is not an int?")
        logger.info("running on processor mesh={}".format(mesh))
        self.domain = de.Domain([y_basis, z_basis, x_basis], grid_dtype=np.float64, mesh=mesh)
        self.Lx = Lx
        self.ar = ar
        self.Ny = Ny
        self.Nz = Nz
        self.Nx = Nx

    def build_var_hotel(self, ignore=[]):
        self.hotel = OrderedDict()
        shape = [Nt]
        for basis in self.domain.basis:
            shape.append(basis.base_grid_size)
        shape = tuple(shape)
        for var in self.forward_problem.variables:
            if (var in ignore):
                continue
            self.hotel[var] = np.zeros(shape)

        

    def build_loop(self, b0, u0):
        self.forward_problem = Forward_problem(self)
        self.backward_problem = Backward_problem()
        b0 = IC_factory.build_ic(self.domain, 0)
        u0 = IC_factory.build_ic(self.domain, 0)
        self.opt_loop = Optimization_loop(self, 0, forward_problem, backward_problem, b0, u0)

    def loop():
        # # forward
        # self.forward_solver().step()
        

        # backward t2 --> t1
        self.forward_solver().step()
        for k in self.hotel.keys():
            self.hotel[k][t_ind] = self.forward_solver.state[k]['g']

        self.backward_solver().step()

        # remeber to set t_ind = 0 at end dummy





class Optimization_loop:
    def __init__(self, opt_context, Nloop, b0, u0):
        return
    
    



class Simulation_params:
    optimization_params



    


