from contextlib import nullcontext
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
from Forward_problem import Forward_problem
from collections import OrderedDict

class OptimizationContext:
    def __init__(self, forward_problem, backward_problem, lagrangian_dict, opt_params, sim_params, timestepper, write_suffix):
        self.forward_problem = forward_problem
        self.backward_problem = backward_problem
        self.lagrangian_dict = lagrangian_dict
        self.forward_solver = forward_problem.build_solver(timestepper)
        self.backward_solver = None
        self.domain = forward_problem.domain
        self.opt_params = opt_params
        self.sim_params = sim_params
        self.timestepper = timestepper
        self.write_suffix = write_suffix
        self.ic = FieldSystem([self.domain.new_field(name=var) for var in forward_problem.variables])
        # self.fc = FieldSystem([self.domain.new_field(name=var) for var in backward_problem.variables])
        # self.integrand_ = np.zeros(opt_params.T // opt_params.dt)
        self.loop_index = 0

    def build_integrator(self, time_dependent_integrad):
        self.backward_problem.append('integrad')
        self.backward_problem.add_equation('integrad = ' + time_dependent_integrad)
        self.backward_solver = self.backward_problem.build_solver(self.timestepper)

    def build_var_hotel(self):
        self.hotel = OrderedDict()
        shape = [self.opt_params.dt_per_cp]
        for basis in self.domain.basis:
            shape.append(basis.base_grid_size)
        shape = tuple(shape)
        for var in self.forward_problem.variables:
            # if (var in self.backward_problem.parameters):
            self.hotel[var] = np.zeros(shape)

    def set_forward_ic(self):
        for var in self.forward_problem.variables:
            if (var in self.ic.field_dict.keys()):
                self.forward_solver.state[var]['c'] = self.ic.field_dict[var]['c']

    def set_backward_fc(self):
        for forward_var in self.lagrangian_dict.keys():
            backward_var, fc_func = self.lagrangian_dict[forward_var]
            self.backward_solver.state[backward_var]['c'] = fc_func(self.forward_solver.state[forward_var]['c'])


    def loop(self): # move to main
        self.set_forward_ic()
        self.solve_forward()
        self.evaluate_state_T()
        self.set_backward_fc()
        self.solve_backward()
        self.loop_index += 1

    def solve_backward(self):
        for cp_index in range(self.opt_params.num_cp):
            # load checkpoint for ic
            self.forward_solver.load_state(self.write_suffix + '_loop_cps.h5', -cp_index - 1)

            for t_ind in range(self.opt_params.dt_per_cp):
                self.forward_solver().step(self.opt_params.dt)
                for var in self.hotel.keys():
                    self.hotel[var][t_ind] = self.forward_solver.state[var]['g']

            for t_ind in range(self.opt_params.dt_per_cp):
                for var in self.hotel.keys():
                    self.backward_problem.parameters[var]['g'] = self.hotel[var][-t_ind - 1]
                
                self.backward_solver().step(self.opt_params.dt)
                # self.integrand_array[cp_index * self.opt_params.dt_per_cp + t_ind] = self.backward_problem.state['integrand']['g']

    # def solve_forward(self):
    #     checkpoints = self.forward_solver.evaluator.add_file_handler('checkpoints_' + self.write_suffix, sim_dt=self.opt_params.dTcp, max_writes=10, mode='overwrite')
    #     checkpoints.add_system(self.forward_solver.state)
    #     self.forward_solver.stop_sim_time = self.opt_params.T
    #     try:
    #         logger.info('Starting forward solve')
    #         while self.forward_solver.ok:
    #             self.forward_solver.step(self.opt_params.dt)
    #     except:
    #         logger.error('Exception raised in forward solve, triggering end of main loop.')
    #         raise
    #     finally:
    #         logger.info('Completed forward solve')

    def evaluate_state_T(self):
        return
  

    def descend(self):
        return

    


