from contextlib import nullcontext
from typing import OrderedDict
import numpy as np
import os
import sys
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import matplotlib.pyplot as plt

import logging
logging.getLogger('solvers').setLevel(logging.INFO)
logger = logging.getLogger(__name__)
from collections import OrderedDict

class OptimizationContext:
    def __init__(self, domain, coords, forward_solver, backward_solver, lagrangian_dict, sim_params, write_suffix):
        
        self.forward_solver = forward_solver
        self.backward_solver = backward_solver
        self.forward_problem = forward_solver.problem
        self.backward_problem = backward_solver.problem
        self.lagrangian_dict = lagrangian_dict
        self.domain = domain
        self.coords = coords
        self.sim_params = sim_params
        self.write_suffix = write_suffix
        
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.ic = OrderedDict()
        for var_name in lagrangian_dict.keys():
            self.ic[var_name] = self.domain.dist.VectorField(coords, bases=domain.bases)
        self.backward_ic = OrderedDict()

        self.loop_index = 0
        self.step_performance = np.nan
        self.HT_norms = []
        self.indices = []

        self.use_euler = True
        self.show = False
        self.show_backward = False
        self.show_cadence = 1

        self.new_x = forward_solver.state[0].copy()
        self.new_grad = forward_solver.state[0].copy()
        self.old_x = forward_solver.state[0].copy()
        self.old_grad = forward_solver.state[0].copy()

        self.new_x.name = "new_x"
        self.new_grad.name = "new_grad"
        self.old_x.name = "old_x"
        self.old_grad.name = "old_grad"

        self.new_x['g'] = 0.0
        self.new_grad['g'] = 0.0
        self.old_x['g'] = 0.0
        self.old_grad['g'] = 0.0

    def set_time_domain(self, T, num_cp, dt):
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

    # Hotel stores the forward variables, at each timestep, in memory to inform adjoint solve
    def build_var_hotel(self):
        self.hotel = OrderedDict()
        # shape = [self.dt_per_cp]
        grid_shape = self.ic['u']['g'].shape
        grid_time_shape = (self.dt_per_cp,) + grid_shape
        for var in self.forward_problem.variables:
            if (var.name in self.lagrangian_dict.keys()):
                self.hotel[var.name] = np.zeros(grid_time_shape)

    def loop(self): # move to main
        
        # Grab before fields are evolved (old state)
        self.old_grad['g'] = self.backward_solver.state[0]['g'].copy()

        self.set_forward_ic()
        
        # Nothing happens here if self.num_cp = 1
        self.solve_forward_full()
        self.forward_solver.evaluator.handlers.clear()
        
        # self.solve_forward()
        self.evaluate_state_T()
        self.set_backward_ic()
        self.solve_backward()

        for i in range(1, self.num_cp):
            self.forward_solver.load_state(self.path + '/checkpoints_' + self.write_suffix + '/checkpoints_kdv0_s1.h5', -i)
            self.solve_forward()
            self.solve_backward()


        # Evaluate after fields are evolved (new state)
        self.backward_solver.state[0].change_scales(1)
        self.new_grad.change_scales(1)
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
        
        self.forward_solver.iteration = 0
        self.forward_solver.stop_sim_time = self.T

        checkpoints = self.forward_solver.evaluator.add_file_handler(self.path + '/checkpoints_{}'.format(self.write_suffix), max_writes=self.num_cp - 1, iter=self.dt_per_cp, mode='overwrite')
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
            for t_ind in range(self.dt_per_loop):

                self.forward_solver.step(self.dt)
                if (t_ind >= self.dt_per_loop - self.dt_per_cp):
                    for var in self.forward_solver.state:
                        if (var.name in self.hotel.keys()):
                            var.change_scales(1)
                            self.hotel[var.name][t_ind - (self.dt_per_loop - self.dt_per_cp)] = var['g'].copy()

                if self.show and t_ind % self.show_cadence == 0:
                    u.change_scales(1)
                    p.set_ydata(u['g'])
                    plt.pause(5e-3)
                    fig.canvas.draw()
                    logger.debug('Full Forward solver: sim_time = {}'.format(self.forward_solver.sim_time))

        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            plt.close()
            logger.debug('Completed forward solve')

    def solve_forward(self):

        # Main loop
        try:
            logger.debug('Starting forward solve')
            for t_ind in range(self.dt_per_cp):
                self.forward_solver.step(self.dt)
                for var in self.forward_solver.state:
                    if (var.name in self.hotel.keys()):
                        var.change_scales(1)
                        self.hotel[var.name][t_ind] = var['g'].copy()
                logger.debug('Forward solver: sim_time = {}'.format(self.forward_solver.sim_time))

        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            logger.debug('Completed forward solve')

    # Set ic for adjoint problem for loop
    def set_backward_ic(self):

        self.backward_solver.sim_time = self.T

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
        try:
            logger.debug('Starting backward solve')
            for t_ind in range(self.dt_per_cp):
                for var in self.hotel.keys():
                    self.backward_solver.problem.namespace[var].change_scales(1)
                    self.backward_solver.problem.namespace[var]['g'] = self.hotel[var][-t_ind - 1]
                self.backward_solver.step(-self.dt)
                logger.debug('backward solver time = {}'.format(self.backward_solver.sim_time))
        except:
            logger.error('Exception raised in backward solve, triggering end of main loop.')
            raise
        finally:
            logger.debug('Completed backward solve')
            for field in self.backward_solver.state:
                field.change_scales(1)

    def evaluate_state_T(self):

        grad_mag = (d3.Integrate((self.new_grad**2))**(0.5)).evaluate()
        HT_norm = d3.Integrate(self.HT).evaluate()

        if (self.loop_index > 0):
            old_HT_norm = self.HT_norm
            gamma = self.gamma

        if (CW.rank == 0):
            self.HT_norm = HT_norm['g'].flat[0]
            self.grad_norm = grad_mag['g'].flat[0]
        else:
            self.HT_norm = 0.0
            self.grad_norm = 0.0

        self.HT_norm = CW.bcast(self.HT_norm, root=0)
        self.grad_norm = CW.bcast(self.grad_norm, root=0)

        if (np.isnan(self.HT_norm)):
            logger.error("NaN HT norm: exiting...")
            sys.exit()

        if (self.loop_index > 0):
            self.step_performance = (old_HT_norm - self.HT_norm) / (self.grad_norm**2 * gamma)
        return
   
   # This is work really well for periodic kdv
    def compute_gamma(self, epsilon_safety):
        if (self.loop_index == 0):
            return 1e-3
        else:
            # https://en.wikipedia.org/wiki/Gradient_descent
            grad_diff = self.new_grad - self.old_grad
            x_diff = self.new_x - self.old_x
            return epsilon_safety * np.abs(d3.Integrate(x_diff * grad_diff).evaluate()['g'].flat[0]) / (d3.Integrate(grad_diff * grad_diff).evaluate()['g'].flat[0])


    def descend(self, gamma, **kwargs):

        addendum_str = kwargs.get('addendum_str', '')

        # This can probably go elsewhere (whereever it's getting dealiased)
        self.new_x.change_scales(1)
        self.old_x.change_scales(1)
        self.new_grad.change_scales(1)
        self.old_grad.change_scales(1)

        self.gamma = gamma
        self.old_x['g'] = self.ic['u']['g'].copy()

        if (self.loop_index == 0 or self.use_euler):
            deltaIC = gamma * self.backward_solver.state[0]['g'].copy()
        
        else:
            #Todo: implement conjugate gradient schemes: https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method

            # 2nd-order Adams Bashforth (nonuniform step)
            # this doesn't work so well
            h0 = self.old_gamma
            h1 = gamma
            y0prime = self.old_grad['g'].copy()
            y1prime = self.new_grad['g'].copy()
            deltaIC = (h1 + h1**2 / 2 / h0) * y1prime - h1**2 / 2 / h0 * y0prime

        self.ic['u']['g'] = self.ic['u']['g'].copy() + deltaIC
        self.new_x['g'] = self.ic['u']['g'].copy()

        statement = 'loop index = %i; ' %self.loop_index
        statement += 'gamma = %e; ' %self.gamma
        statement += 'grad norm = %e; ' %self.grad_norm
        statement += 'step performance = %f; ' %self.step_performance
        statement += 'HT norm = %e; ' %self.HT_norm
        statement += addendum_str

        logger.info(statement)
        self.loop_index += 1

        self.indices.append(self.loop_index)
        self.HT_norms.append(self.HT_norm)

        self.old_gamma = gamma

    def richardson_gamma(self, gamma):

        # logger.info("Performing Richardson extrapolation to measure gradient magnitude linearity...")
        self.loop()
        HT_norm_og = self.HT_norm

        # Richardson loop 1: descend IC by a small amount and repeat.
        self.descend(gamma)
        self.loop()
        delta_HT1 = HT_norm_og - self.HT_norm

        # Richardson loop 2: repeating...
        self.descend(gamma)
        self.loop()
        delta_HT2 = HT_norm_og - self.HT_norm

        # We expect, for sufficiently small gamma, the objective (HT_norm) to change by an equal amount both times
        linearity = delta_HT2 / delta_HT1 / 2.0
        return linearity


    def update_timestep(self, dt):
        if (self.dT / dt != int(self.dT / dt)):
            logger.error("number of timesteps not divisible by number of checkpoints. cannot update dt...")
            return
        self.dt = dt
        self.dt_per_cp = int(self.dT // dt)
        self.build_var_hotel()
