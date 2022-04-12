from contextlib import nullcontext
from typing import OrderedDict
import numpy as np
import os
import sys
import h5py
import time
import pickle
import dedalus.public as d3
from dedalus.core.system import FieldSystem
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import plot_bot_2d
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logging.getLogger('solvers').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
from collections import OrderedDict
import matplotlib.pyplot as plt

class OptParams:
    def __init__(self, T, num_cp, dt):
        self.T = T
        self.num_cp = num_cp
        self.dt = dt
        self.dT = T / num_cp
        if (self.dT / dt != int(self.dT / dt)):
            logger.error("number of timesteps not divisible by number of checkpoints. Exiting...")
            sys.exit()
        if (self.T / dt != int(self.T / dt)):
            logger.error("Run period not divisible by timestep (we're using uniform timesteps). Exiting...")
            sys.exit()
        self.dt_per_cp = int(self.dT / dt)
        self.dt_per_loop = int(T / dt)

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
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.ic = OrderedDict()
        for var_name in lagrangian_dict.keys():
            self.ic[var_name] = self.domain.dist.VectorField(coords, bases=domain.bases)
        self.backward_ic = OrderedDict()
        self.loop_index = 0
        self.show = False
        self.show_backward = False

        self.new_x = domain.dist.Field(name='new_x', bases=domain.bases)
        self.new_grad = domain.dist.Field(name='new_grad', bases=domain.bases)
        self.old_x = domain.dist.Field(name='old_x', bases=domain.bases)
        self.old_grad = domain.dist.Field(name='old_grad', bases=domain.bases)

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
        
        # Evaluate before fields are evolved (old state)
        self.old_grad['g'] = self.backward_solver.state[0]['g'].copy()

        self.set_forward_ic()
        
        # if (self.opt_params.num_cp > 1.0):

        self.solve_forward_full()
        self.forward_solver.evaluator.handlers.clear()
        
        self.solve_forward()
        self.evaluate_state_T()
        self.set_backward_ic()
        self.solve_backward()

        for i in range(1, self.opt_params.num_cp):
            self.forward_solver.load_state(self.path + '/checkpoints_' + self.write_suffix + '/checkpoints_kdv0_s1.h5', -i)
            self.solve_forward()
            self.solve_backward()


        # Evaluate after fields are evolved (new state)
        self.new_grad['g'] = self.backward_solver.state[0]['g'].copy()

    # Set starting point for loop
    def set_forward_ic(self):
        # self.forward_solver = self.forward_problem.build_solver(self.timestepper) 

        self.forward_solver.sim_time = 0.0
        self.forward_solver.iteration = 0
        for var in self.forward_solver.state:
            if (var.name in self.ic.keys()):
                var.change_scales(1)
                var['g'] = self.ic[var.name]['g']

    def solve_forward_full(self):
        self.forward_solver.stop_sim_time = self.opt_params.T
        self.forward_solver.iteration = 0
        if (self.opt_params.num_cp == 1):
            return

        checkpoints = self.forward_solver.evaluator.add_file_handler(self.path + '/checkpoints_{}'.format(self.write_suffix), max_writes=self.opt_params.num_cp - 1, iter=self.opt_params.dt_per_cp, mode='overwrite')
        checkpoints.add_tasks(self.forward_solver.state, layout='g')

        # Main loop
        if (self.show):
            u = self.forward_solver.state[0]
            u.change_scales(1)
            fig = plt.figure()
            p, = plt.plot(self.x, u['g'])
            plt.plot(self.x, self.U_data)
            plt.title('Loop Index = {}'.format(self.loop_index))
            fig.canvas.draw()
        try:
            logger.debug('Starting forward solve')
            for t_ind in range(self.opt_params.dt_per_loop - self.opt_params.dt_per_cp):

                self.forward_solver.step(self.opt_params.dt)

                # if (t_ind > 0 and (t_ind + 1) % self.opt_params.dt_per_cp == 0):
                #     self.forward_solver.evaluator.evaluate_handlers(self.forward_solver.evaluator.handlers, wall_time=self.forward_solver.stop_sim_time, sim_time=self.forward_solver.sim_time, iteration=self.forward_solver.iteration)

                if self.show and t_ind % 25 == 0:
                    u.change_scales(1)
                    p.set_ydata(u['g'])
                    plt.pause(5e-3)
                    fig.canvas.draw()
                # logger.info('Forward solver: sim_time = {}'.format(self.forward_solver.sim_time))

            # for var in self.forward_solver.state:
            #     if (var.name in self.hotel.keys()):
            #         var.change_scales(1)
            #         self.hotel[var.name][self.opt_params.dt_per_cp] = var['g'].copy()
        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            plt.close()
            logger.debug('Completed forward solve')

    def solve_forward(self):
        self.forward_solver.stop_sim_time = self.opt_params.T

        # Main loop
        if (self.show):
            u = self.forward_solver.state[0]
            u.change_scales(1)
            fig = plt.figure()
            p, = plt.plot(self.x, u['g'])
            plt.plot(self.x, self.U_data)
            plt.title('Loop Index = {}'.format(self.loop_index))
            fig.canvas.draw()
        try:
            logger.debug('Starting forward solve')
            for t_ind in range(self.opt_params.dt_per_cp):
                for var in self.forward_solver.state:
                    if (var.name in self.hotel.keys()):
                        var.change_scales(1)
                        self.hotel[var.name][t_ind] = var['g'].copy()
                self.forward_solver.step(self.opt_params.dt)
                if self.show and t_ind % 25 == 0:
                    u.change_scales(1)
                    p.set_ydata(u['g'])
                    plt.pause(5e-3)
                    fig.canvas.draw()
                # logger.info('Forward solver: sim_time = {}'.format(self.forward_solver.sim_time))

            for var in self.forward_solver.state:
                if (var.name in self.hotel.keys()):
                    var.change_scales(1)
                    self.hotel[var.name][self.opt_params.dt_per_cp] = var['g'].copy()
        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            plt.close()
            logger.debug('Completed forward solve')

    # Set ic for adjoint problem for loop
    def set_backward_ic(self):

        # self.backward_solver = self.backward_problem.build_solver(self.timestepper)
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

    def solve_backward(self):
        # self.backward_solver.stop_sim_time = self.opt_params.T
        if (self.show_backward):
            u_t = self.backward_solver.state[0]
            u_t.change_scales(1)
            fig = plt.figure()
            p, = plt.plot(self.x, u_t['g'])
            # plt.plot(self.x, self.U_data)
            plt.title('Loop Index = {}'.format(self.loop_index))
            fig.canvas.draw()
        try:
            logger.debug('Starting backward solve')
            for t_ind in range(self.opt_params.dt_per_cp):
                for var in self.hotel.keys():
                    self.backward_solver.problem.namespace[var].change_scales(1)
                    self.backward_solver.problem.namespace[var]['g'] = self.hotel[var][-t_ind - 1]
                if self.show_backward and (t_ind % 50) == 0:
                    u_t.change_scales(1)
                    p.set_ydata(u_t['g'])
                    plt.pause(5e-3)
                    fig.canvas.draw()
                    # logger.info('Backward solver: sim_time = {}; u_t = {}'.format(self.backward_solver.sim_time, np.max(self.backward_solver.state[0]['g'])))
                self.backward_solver.step(-self.opt_params.dt)
        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            logger.debug('Completed backward solve')
            for field in self.backward_solver.state:
                field.change_scales(1)

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
        self.HT_norm = d3.Integrate(self.HT).evaluate()['g'][0]
        if (np.isnan(self.HT_norm)):
            logger.error("NaN HT norm: exiting...")
            sys.exit()
        return
  
    def compute_gamma(self, epsilon_safety):
        if (self.loop_index == 0):
            return 0.0001
        else:
            # https://en.wikipedia.org/wiki/Gradient_descent
            grad_diff = self.new_grad - self.old_grad
            x_diff = self.new_x - self.old_x
            return epsilon_safety * np.abs(d3.Integrate(x_diff * grad_diff).evaluate()['g'][0]) / (d3.Integrate(grad_diff * grad_diff).evaluate()['g'][0])


    def descend(self, gamma):

        # This can probably go elsewhere (where it's getting dealiased)
        self.new_x.change_scales(1)
        self.old_x.change_scales(1)
        self.new_grad.change_scales(1)
        self.old_grad.change_scales(1)
        self.backward_solver.state[0].change_scales(1)

        self.old_x['g'] = self.ic['u']['g'].copy()

        if (self.loop_index == 0 or self.use_euler):
            deltaIC = gamma * self.backward_solver.state[0]['g'].copy()
        
        else:

            # # 2nd-order Adams Bashforth (nonuniform step)
            h0 = self.old_gamma
            h1 = gamma
            y0prime = self.old_grad['g'].copy()
            y1prime = self.new_grad['g'].copy()
            # deltaIC = gamma * y1prime

            deltaIC = (h1 + h1**2 / 2 / h0) * y1prime - h1**2 / 2 / h0 * y0prime

        self.ic['u']['g'] = self.ic['u']['g'].copy() + deltaIC
        self.new_x['g'] = self.ic['u']['g'].copy()
        
        logger.info('loop index = %i; gamma = %e; HT norm = %e; ' %(self.loop_index, gamma, self.HT_norm))
        self.loop_index += 1

        self.old_gamma = gamma

    def update_timestep(self, dt):
        if (self.opt_params.dT / dt != int(self.opt_params.dT / dt)):
            logger.error("number of timesteps not divisible by number of checkpoints. cannot update dt...")
            return
        self.opt_params.dt = dt
        self.opt_params.dt_per_cp = int(self.opt_params.dT // dt)
        self.build_var_hotel()
