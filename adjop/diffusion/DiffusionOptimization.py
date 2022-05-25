from distutils.command.bdist import show_formats
import os
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(path + "/..")
from OptimizationContext import OptimizationContext
import h5py
import gc
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
import ForwardDiffusion
import BackwardDiffusion
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime

class DiffusionOptimization(OptimizationContext):

    def project_ic(self):
        u = self.ic['u']
        uf = self.domain.dist.Field(name='uf', bases=self.domain.bases)
        uf['g'] = (u['g'][0, :].copy())**2
        int1 = (uf*self.w1).evaluate()
        int2 = (uf*self.w2).evaluate()
        self.x1.append(d3.Integrate(int1, 'x').evaluate()['g'].flat[0])
        self.x2.append(d3.Integrate(int2, 'x').evaluate()['g'].flat[0])

    def before_fullforward_solve(self):
        self.project_ic()
        if (self.loop_index % self.show_loop_cadence == 0 and self.show):
            u = self.forward_solver.state[0]
            u.change_scales(1)
            self.fig = plt.figure()
            self.p, = plt.plot(self.x_grid, u['g'])
            plt.plot(self.x_grid, self.U_data)
            self.fig.canvas.draw()
            title = plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.forward_solver.sim_time, 1)))
            # plt.show()
                
        logger.debug('start evaluating f..')

    def during_fullforward_solve(self):
        if (self.loop_index % self.show_loop_cadence == 0 and self.show and self.forward_solver.iteration % self.show_iter_cadence == 0):
            u = self.forward_solver.state[0]
            u.change_scales(1)
            self.p.set_ydata(u['g'])
            plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.forward_solver.sim_time, 1)))
            plt.pause(1e-10)
            self.fig.canvas.draw()

    def after_fullforward_solve(self):
        loop_message = 'loop index = {}; '.format(self.loop_index)
        loop_message += 'objectiveT = {}; '.format(self.objectiveT_norm)
        for metric_name in self.metricsT_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, self.metricsT_norms[metric_name])
        logger.info(loop_message)
        plt.pause(3e-1)
        plt.close()

    def before_backward_solve(self):
        logger.debug('Starting backward solve')
        if (self.loop_index % self.show_loop_cadence == 0 and self.show):
            u = self.backward_solver.state[0]
            u.change_scales(1)
            self.fig = plt.figure()
            self.p, = plt.plot(self.x_grid, u['g'])
            self.fig.canvas.draw()
            title = plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.backward_solver.sim_time, 1)))

    def during_backward_solve(self):
        if (self.loop_index % self.show_loop_cadence == 0 and self.show and self.backward_solver.iteration % self.show_iter_cadence == 0):
            u = self.backward_solver.state[0]
            u.change_scales(1)
            self.p.set_ydata(u['g'])
            plt.title('loop index = {}; t = {}'.format(self.loop_index, round(self.backward_solver.sim_time, 1)))
            plt.pause(1e-10)
            self.fig.canvas.draw()
        # logger.debug('backward solver time = {}'.format(self.backward_solver.sim_time))
        pass

    def after_backward_solve(self):
        plt.pause(3e-1)
        plt.close()
        logger.debug('Completed backward solve')
        pass