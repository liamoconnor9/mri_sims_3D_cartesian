from contextlib import nullcontext
from typing import OrderedDict
import numpy as np
import os
import sys
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import matplotlib.pyplot as plt
import inspect
import logging
logging.getLogger('solvers').setLevel(logging.ERROR)
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
        self.slices = self.domain.dist.grid_layout.slices(self.domain, scales=1)
        self.coords = coords
        self.sim_params = sim_params
        self.write_suffix = write_suffix
        
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.run_dir = os.path.dirname(os.path.abspath(inspect.getmodule(inspect.stack()[1][0]).__file__))
        if not os.path.isdir(self.run_dir + '/' + self.write_suffix):
            logger.info('Creating run directory {}'.format(self.run_dir + '/' + self.write_suffix))
            os.makedirs(self.run_dir + '/' + self.write_suffix)

        self.ic = OrderedDict()
        for var in lagrangian_dict.keys():
            self.ic[var.name] = self.domain.dist.VectorField(coords, bases=domain.bases)
        self.backward_ic = OrderedDict()

        self.loop_index = 0
        self.opt_iters = np.inf
        self.step_performance = np.nan
        self.metricsT = {}
        self.metricsT_norms = {}
        self.objectiveT_norms = []
        self.indices = []

        # self.add_handlers = False
        self.show = False
        self.show_backward = False
        self.gamma_init = 0.01

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
        if (not np.allclose(round(self.dT / dt), self.dT / dt)):
            logger.error("number of timesteps not divisible by number of checkpoints. Exiting...")
        if (not np.allclose(round(self.T / dt), self.T / dt)):
            logger.error("Run period not divisible by timestep (we're using uniform timesteps). Exiting...")
        self.dt_per_cp = round(self.dT / dt)
        self.dt_per_loop = round(T / dt)
        self.build_var_hotel()

    # Hotel stores the forward variables, at each timestep, in memory to inform adjoint solve
    def build_var_hotel(self):
        self.hotel = OrderedDict()
        # shape = [self.dt_per_cp]
        grid_shape = self.ic['u']['g'].shape
        grid_time_shape = (self.dt_per_cp,) + grid_shape
        for var in self.forward_problem.variables:
            if (var in self.lagrangian_dict.keys()):
                self.hotel[var.name] = np.zeros(grid_time_shape)

    def set_objectiveT(self, objectiveT):
        self.objectiveT = objectiveT
        self.backward_ic = OrderedDict()
        for forward_field in self.lagrangian_dict.keys():
            backward_field = self.lagrangian_dict[forward_field]
            self.backward_ic[backward_field.name] = objectiveT.sym_diff(forward_field)

    def reshape_soln(self, x):
        return x.reshape(self.domain.grid_shape(scales=1))[:]

    # For a given problem, these should be overwritten to add filehandlers, animations, metrics, etc.
    def before_fullforward_solve(self):
        pass

    def after_fullforward_solve(self):
        pass

    def before_backward_solve(self):
        pass

    def during_backward_solve(self):
        pass

    def after_backward_solve(self):
        pass

    # Depreciated with scipy.optimization.minimize
    # calling this solves for the objective (forward) and the gradient (backward)
    def loop(self): 
        self.loop_forward()
        self.loop_backward()

    def loop_forward(self, x):

        if (self.loop_index >= self.opt_iters):
            raise self.LoopIndexException({"message": "Achieved the proper number of loop index"})

        self.x = x
        self.ic['u'].change_scales(1)
        self.old_x.change_scales(1)
        self.new_x.change_scales(1)
        # self.ic['u']['g'] = x.reshape((2,) + self.domain.grid_shape(scales=1))[:, self.slices[0], self.slices[1]]

        self.old_x['g'] = self.ic['u']['g'].copy()
        self.ic['u']['g'] = self.reshape_soln(x)
        self.new_x['g'] = self.ic['u']['g'].copy()

        self.before_fullforward_solve()
        
        # Grab before fields are evolved (old state)
        self.old_grad.change_scales(1)
        self.backward_solver.state[0].change_scales(1)
        self.old_grad['g'] = self.backward_solver.state[0]['g'].copy()

        self.set_forward_ic()       

        self.solve_forward_full()

        self.forward_solver.evaluator.handlers.clear()
        
        # self.solve_forward()
        self.evaluate_objectiveT()
        self.objectiveT_norms.append(self.objectiveT_norm)
        self.indices.append(self.loop_index)
        self.after_fullforward_solve()

        if (self.objectiveT_norm <= min(self.objectiveT_norms)):
            self.best_x = x
            self.best_index = self.loop_index
            self.best_objectiveT = self.objectiveT_norm

        return self.objectiveT_norm

    def loop_backward(self, x):

        if not np.allclose(x, self.x):
            logger.warning('Repeating forward solver without computing gradient. This is probably inefficient!!')
            self.loop_forward(x)

        self.set_backward_ic()
        self.before_backward_solve()
        self.solve_backward()

        for i in range(1, self.num_cp):
            self.forward_solver.load_state(self.run_dir + '/' + self.write_suffix + '/checkpoints_internal/checkpoints_internal_s1.h5', -i)
            self.solve_forward()
            self.solve_backward()

        self.backward_solver.evaluator.handlers.clear()

        # Evaluate after fields are evolved (new state)
        self.backward_solver.state[0].change_scales(1)
        self.backward_solver.state[0]['g']

        self.old_grad.change_scales(1)
        self.new_grad.change_scales(1)

        self.old_grad['g'] = self.new_grad['g'].copy()
        self.new_grad['g'] = self.backward_solver.state[0]['g'].copy()

        self.after_backward_solve()
        self.loop_index += 1

        return self.gamma_init * self.backward_solver.state[0].allgather_data().flatten().copy()
        
    # Set starting point for loop
    def set_forward_ic(self):
        self.forward_solver.sim_time = 0.0
        self.forward_solver.iteration = 0
        for var in self.forward_solver.state:
            if (var.name in self.ic.keys()):
                var.change_scales(1)
                ic = self.ic[var.name].evaluate()
                ic.change_scales(1)
                var['g'] = ic['g'].copy()

    def solve_forward_full(self):

        solver = self.forward_solver
        solver.iteration = 0
        solver.stop_sim_time = self.T

        checkpoints = solver.evaluator.add_file_handler(self.run_dir + '/' + self.write_suffix + '/checkpoints_internal', max_writes=self.num_cp - 1, iter=self.dt_per_cp, mode='overwrite')
        checkpoints.add_tasks(solver.state, layout='g')

        # Main loop
        # if (self.show):
        #     u = solver.state[0]
        #     u.change_scales(1)
        #     fig = plt.figure()
        #     p, = plt.plot(self.x, u['g'])
        #     plt.plot(self.x, self.U_data)
        #     plt.title('Loop Index = {}'.format(self.loop_index))
        #     fig.canvas.draw()
        try:
            logger.debug('Starting forward solve')
            for t_ind in range(self.dt_per_loop):

                solver.step(self.dt)
                if (t_ind >= self.dt_per_loop - self.dt_per_cp):
                    for var in solver.state:
                        if (var.name in self.hotel.keys()):
                            var.change_scales(1)
                            self.hotel[var.name][t_ind - (self.dt_per_loop - self.dt_per_cp)] = var['g'].copy()
                self.during_fullforward_solve()

                # if self.show and t_ind % self.show_cadence == 0:
                #     u.change_scales(1)
                #     p.set_ydata(u['g'])
                #     plt.pause(5e-3)
                #     fig.canvas.draw()

        except:
            logger.error('Exception raised in forward solve, triggering end of main loop.')
            raise
        finally:
            # plt.close()
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
            if (backward_field.name in self.backward_ic.keys()):
                backward_ic_field = self.backward_ic[backward_field.name].evaluate()
                backward_field.change_scales(1)
                backward_ic_field.change_scales(1)
                backward_field['g'] = backward_ic_field['g'].copy()
        return

    def solve_backward(self):
        try:
            for t_ind in range(self.dt_per_cp):
                for var in self.hotel.keys():
                    self.backward_solver.problem.namespace[var].change_scales(1)
                    self.backward_solver.problem.namespace[var]['g'] = self.hotel[var][-t_ind - 1]
                self.backward_solver.step(-self.dt)
                self.during_backward_solve()
        except:
            logger.error('Exception raised in backward solve, triggering end of main loop.')
            raise
        finally:
            for field in self.backward_solver.state:
                field.change_scales(1)

    def evaluate_objectiveT(self):

        objectiveT_norm = d3.Integrate(self.objectiveT).evaluate()

        if (CW.rank == 0):
            self.objectiveT_norm = objectiveT_norm['g'].flat[0]
        else:
            self.objectiveT_norm = 0.0

        self.objectiveT_norm = CW.bcast(self.objectiveT_norm, root=0)

        if (np.isnan(self.objectiveT_norm)):
            raise self.NanNormException({"message": "NaN objectiveT_norm computed. Ending optimization loop..."})

        for metric_name in self.metricsT.keys():
            metricT_norm = d3.Integrate(self.metricsT[metric_name]).evaluate()

            if (CW.rank == 0):
                self.metricsT_norms[metric_name] = metricT_norm['g'].flat[0]
            else:
                self.metricsT_norms[metric_name] = 0.0

            self.metricsT_norms[metric_name] = CW.bcast(self.metricsT_norms[metric_name], root=0)

        return

#     def evaluate_initial_state(self):

#         grad_mag = (d3.Integrate((self.new_grad**2))**(0.5)).evaluate()
#         graddiff_mag = (d3.Integrate((self.old_grad*self.new_grad))**(0.5)).evaluate()

#         if (CW.rank == 0):
#             self.grad_norm = grad_mag['g'].flat[0]
#             self.graddiff_norm = graddiff_mag['g'].flat[0]
#         else:
#             self.grad_norm = 0.0
#             self.graddiff_norm = 0.0

#         self.grad_norm = CW.bcast(self.grad_norm, root=0)
#         self.graddiff_norm = CW.bcast(self.graddiff_norm, root=0)

#         return

   # This works really well for periodic kdv
    def compute_gamma(self, epsilon_safety):
        if (self.loop_index == 1):
            return 1e-3
        else:
            # https://en.wikipedia.org/wiki/Gradient_descent
            grad_diff = self.new_grad - self.old_grad
            x_diff = self.new_x - self.old_x
            integ1 = d3.Integrate(x_diff * grad_diff).evaluate()
            integ2 = d3.Integrate(grad_diff * grad_diff).evaluate()

            old_grad_sqrd_integ = d3.Integrate(self.old_grad * self.old_grad).evaluate()
            if (CW.rank == 0):
                gamma = epsilon_safety * np.abs(integ1['g'].flat[0]) / (integ2['g'].flat[0])
                old_grad_sqrd = old_grad_sqrd_integ['g'].flat[0]
            else:
                gamma = 0.0
                old_grad_sqrd = 0.0
            gamma = CW.bcast(gamma, root=0)
            self.old_grad_sqrd = CW.bcast(old_grad_sqrd, root=0)

            return gamma

#     def descend(self, gamma, **kwargs):

#         addendum_str = kwargs.get('addendum_str', '')

#         # This can probably go elsewhere (whereever it's getting dealiased)
#         self.new_x.change_scales(1)
#         self.old_x.change_scales(1)
#         self.new_grad.change_scales(1)
#         self.old_grad.change_scales(1)

#         self.gamma = gamma
#         list(self.ic.items())[0][1].change_scales(1)
#         self.old_x['g'] = list(self.ic.items())[0][1]['g'].copy()

#         self.deltaIC = self.backward_solver.state[0]

#         self.ic['u']['g'] = self.ic['u']['g'].copy() + gamma * self.deltaIC['g']
#         self.new_x['g'] = self.ic['u']['g'].copy()

#         statement = 'loop index = %i; ' %self.loop_index
#         statement += 'gamma = %e; ' %self.gamma
#         statement += 'grad norm = %e; ' %self.grad_norm
#         statement += 'step performance = %f; ' %self.step_performance
#         statement += 'objectiveT norm = %e; ' %self.objectiveT_norm
#         statement += addendum_str

#         logger.info(statement)
#         self.loop_index += 1

#         self.indices.append(self.loop_index)
#         self.objectiveT_norms.append(self.objectiveT_norm)

    class LoopIndexException(Exception):
        pass

    class NanNormException(Exception):
        pass

    class DescentStallException(Exception):
        pass