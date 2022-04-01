from contextlib import nullcontext
from typing import OrderedDict
import numpy as np
import os
import sys
import h5py
import time
import pickle
import dedalus.public as de
from dedalus.core.system import FieldSystem
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import plot_bot_2d
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
from OptParams import OptParams
from collections import OrderedDict
import matplotlib.pyplot as plt

class OptimizationContext:
    def __init__(self, domain, coords, forward_solver, backward_solver, timestepper, lagrangian_dict, opt_params, sim_params, write_suffix):
        self.forward_solver = forward_solver
        self.backward_solver = backward_solver
        self.forward_problem = forward_solver.problem
        self.backward_problem = backward_solver.problem
        self.timestepper = timestepper
        self.lagrangian_dict = lagrangian_dict
        self.domain = domain
        self.coords = coords
        self.opt_params = opt_params
        self.sim_params = sim_params
        self.write_suffix = write_suffix
        self.ic = OrderedDict()
        for var_name in lagrangian_dict.keys():
            self.ic[var_name] = self.domain.dist.VectorField(coords, bases=domain.bases)
        self.backward_ic = OrderedDict()
        self.loop_index = 0
        self.show = False

    # Hotel stores the forward variables, at each timestep, in memory to inform adjoint solve
    def build_var_hotel(self):
        self.hotel = OrderedDict()
        # shape = [self.opt_params.dt_per_cp]
        grid_shape = self.ic['u']['g'].shape
        grid_time_shape = (self.opt_params.dt_per_cp + 1,) + grid_shape
        for var in self.forward_problem.variables:
            if (var.name in self.lagrangian_dict.keys()):
                self.hotel[var.name] = np.zeros(grid_time_shape)

    def loop(self): # move to main
        self.set_forward_ic()
        self.solve_forward()
        self.evaluate_state_T()
        self.set_backward_ic()
        self.solve_backward()
        logger.info('Loop complete. Loop index = {}'.format(self.loop_index))
        self.loop_index += 1

    # Set starting point for loop
    def set_forward_ic(self):
        self.forward_solver = self.forward_problem.build_solver(self.timestepper) 
        self.forward_solver.sim_time = 0.0
        for var in self.forward_solver.state:
            if (var.name in self.ic.keys()):
                var.change_scales(1)
                var['g'] = self.ic[var.name]['g']

    # Set ic for adjoint problem for loop
    def set_backward_ic(self):

        self.backward_solver = self.backward_problem.build_solver(self.timestepper)
        self.backward_solver.sim_time = self.opt_params.T

        # flip dictionary s.t. keys are backward var names and items are forward var names
        flipped_ld = dict((backward_var, forward_var) for forward_var, backward_var in self.lagrangian_dict.items())
        for backward_field in self.backward_solver.state:
            if (backward_field.name in flipped_ld.keys()):
                field = self.backward_ic[backward_field.name].evaluate()
                field.change_scales(1)
                backward_field.change_scales(1)
                backward_field['g'] = field['g'].copy()
        return

    def solve_forward(self):
        self.forward_solver.stop_sim_time = self.opt_params.T

        # Main loop
        if (self.show):
            u = self.forward_solver.state[0]
            u.change_scales(1)
            fig = plt.figure()
            p, = plt.plot(self.x, u['g'])
            plt.plot(self.x, self.U_data)
            fig.canvas.draw()
        try:
            logger.info('Starting forward solve')
            for t_ind in range(self.opt_params.dt_per_cp):
                self.forward_solver.step(self.opt_params.dt)
                for var in self.forward_solver.state:
                    if (var.name in self.hotel.keys()):
                        var.change_scales(1)
                        self.hotel[var.name][t_ind] = var['g'].copy()
                if self.show and t_ind % 10 == 0:
                    u.change_scales(1)
                    p.set_ydata(u['g'])
                    plt.pause(1e-3)
                    fig.canvas.draw()
                # logger.info('Forward solver: sim_time = {}'.format(self.forward_solver.sim_time))
        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            plt.close()
            logger.info('Completed forward solve')

    def solve_backward(self):
        # self.backward_solver.stop_sim_time = self.opt_params.T
        try:
            logger.info('Starting backward solve')
            for t_ind in range(self.opt_params.dt_per_cp):
                for var in self.hotel.keys():
                    self.backward_solver.problem.namespace[var] = self.hotel[var][-t_ind]
                self.backward_solver.step(-self.opt_params.dt)
        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            logger.info('Completed backward solve')

        # for cp_index in range(self.opt_params.num_cp):
        #     # load checkpoint for ic
        #     self.forward_solver.load_state(self.write_suffix + '_loop_cps.h5', -cp_index - 1)

        #     for t_ind in range(self.opt_params.dt_per_cp):
        #         self.forward_solver().step(self.opt_params.dt)
        #         for var in self.hotel.keys():
        #             self.hotel[var][t_ind] = self.forward_solver.state[var]['g']

        #     for t_ind in range(self.opt_params.dt_per_cp):
        #         for var in self.hotel.keys():
        #             self.backward_problem.parameters[var]['g'] = self.hotel[var][-t_ind - 1]
                
        #         self.backward_solver().step(self.opt_params.dt)
        #         # self.integrand_array[cp_index * self.opt_params.dt_per_cp + t_ind] = self.backward_problem.state['integrand']['g']


    def evaluate_state_T(self):
        self.HT_norm = np.sum(np.abs(self.HT.evaluate()['g']))
        logger.info('HT norm = {}'.format(self.HT_norm))
        return
  

    def descend(self):
        return

    


