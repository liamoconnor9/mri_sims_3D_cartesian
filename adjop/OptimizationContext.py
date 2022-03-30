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

class OptimizationContext:
    def __init__(self, domain, coords, forward_problem, backward_problem, lagrangian_dict, opt_params, sim_params, timestepper, write_suffix):
        self.forward_problem = forward_problem
        self.backward_problem = backward_problem
        self.lagrangian_dict = lagrangian_dict
        self.forward_solver = forward_problem.build_solver(timestepper)
        self.backward_solver = backward_problem.build_solver(timestepper)
        self.domain = domain
        self.coords = coords
        self.opt_params = opt_params
        self.sim_params = sim_params
        self.timestepper = timestepper
        self.write_suffix = write_suffix
        # self.ic = FieldSystem([self.domain.dist.VectorField(coords, name=var, bases=domain.bases) for var in forward_problem.variables], self.forward_solver.subproblems)
        # self.ic = [self.domain.dist.Field() for var in lagrangian_dict.keys()]
        self.ic = OrderedDict()
        for var_name in lagrangian_dict.keys():
            self.ic[var_name] = self.domain.dist.VectorField(coords, bases=domain.bases)
        # self.fc = FieldSystem([self.domain.new_field(name=var) for var in backward_problem.variables])
        self.loop_index = 0

    # Hotel stores the forward variables, at each timestep, in memory to inform adjoint solve
    def build_var_hotel(self):
        self.hotel = OrderedDict()
        # shape = [self.opt_params.dt_per_cp]
        grid_shape = self.ic['u']['g'].shape
        grid_time_shape = (self.opt_params.dt_per_cp + 1,) + grid_shape
        for var in self.forward_problem.variables:
            # if (var in self.backward_problem.parameters):
            if (not 'tau' in var.name):
                self.hotel[var.name] = np.zeros(grid_time_shape)

    # Set starting point for loop
    def set_forward_ic(self):
        for var in self.forward_solver.state:
            if (var.name in self.ic.keys()):
                var['g'] = self.ic[var.name]['g']

    # Set ic for adjoint problem for loop
    def set_backward_fc(self):
        return
        for forward_var in self.lagrangian_dict.keys():
            backward_var, fc_func = self.lagrangian_dict[forward_var]
            self.backward_solver.state[backward_var]['c'] = fc_func(self.forward_solver.state[forward_var]['c'])


    def loop(self): # move to main
        self.set_forward_ic()
        self.solve_forward()
        self.evaluate_state_T()
        self.set_backward_fc()
        self.solve_backward()
        print('success')
        self.loop_index += 1

    def solve_forward(self):
        checkpoints = self.forward_solver.evaluator.add_file_handler('checkpoints_' + self.write_suffix, sim_dt=self.opt_params.dT, max_writes=10, mode='overwrite')
        checkpoints.add_tasks(self.forward_solver.state, layout='c')
        self.forward_solver.stop_sim_time = self.opt_params.T
        try:
            logger.info('Starting forward solve')
            index = 0
            while self.forward_solver.proceed:
                self.forward_solver.step(self.opt_params.dt)
                for var in self.forward_solver.state:
                    if (var.name in self.hotel.keys()):
                        var.change_scales(1)
                        self.hotel[var.name][index] = var['g'].copy()
                index += 1
        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            logger.info('Completed forward solve')

    def solve_backward(self):
        # self.backward_solver.stop_sim_time = self.opt_params.T
        try:
            logger.info('Starting backward solve')
            for t_ind in range(self.opt_params.dt_per_cp):
                for var in self.hotel.keys():
                    self.backward_solver.problem.namespace[var] = self.hotel[var][-t_ind]
                    logger.info('loading state t_ind = {}'.format(t_ind))
                self.backward_solver.step(self.opt_params.dt)
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
        return
  

    def descend(self):
        return

    


