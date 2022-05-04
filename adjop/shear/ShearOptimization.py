"""
Dedalus script for adjoint looping:
Given an end state (checkpoint_U), this script recovers the initial condition with no prior knowledge
Usage:
    shear_cg.py <config_file> <run_suffix>
"""

from distutils.command.bdist import show_formats
import os
path = os.path.dirname(os.path.abspath(__file__))
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import sys
sys.path.append(path + "/..")
import h5py
import gc
import dedalus.public as d3
from dedalus.core import domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
logging.getLogger('solvers').setLevel(logging.ERROR)
# logger.setLevel(logging.info)
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser

from OptimizationContext import OptimizationContext
import ForwardShear
import BackwardShear
import matplotlib.pyplot as plt
from scipy import optimize
from natsort import natsorted

class ShearOptimization(OptimizationContext):
    def reshape_soln(self, x):
        slices = self.slices
        return x.reshape((2,) + self.domain.grid_shape(scales=1))[:, slices[0], slices[1]]

    def before_fullforward_solve(self):
        if self.add_handlers and self.loop_index % self.handler_loop_cadence == 0:
            checkpoints = self.forward_solver.evaluator.add_file_handler(self.run_dir + '/' + self.write_suffix + '/checkpoints/checkpoint_loop'  + str(self.loop_index), max_writes=1, sim_dt=self.T, mode='overwrite')
            checkpoints.add_tasks(self.forward_solver.state, layout='g')
            ex, ez = self.coords.unit_vector_fields(self.domain.dist)

            snapshots = self.forward_solver.evaluator.add_file_handler(self.run_dir + '/' + self.write_suffix + '/snapshots_forward/snapshots_forward_loop' + str(self.loop_index), sim_dt=0.01, max_writes=10, mode='overwrite')
            u = self.forward_solver.state[0]
            s = self.forward_solver.state[1]
            p = self.forward_solver.state[2]
            snapshots.add_task(s, name='tracer')
            snapshots.add_task(p, name='pressure')
            snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
            snapshots.add_task(d3.dot(ex, u), name='ux')
            snapshots.add_task(d3.dot(ez, u), name='uz')

        logger.debug('start evaluating f..')

    def during_fullforward_solve(self):
        pass

    def after_fullforward_solve(self):
        loop_message = 'loop index = {}; '.format(self.loop_index)
        loop_message += 'objectiveT = {}; '.format(self.objectiveT_norm)
        for metric_name in self.metricsT_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, self.metricsT_norms[metric_name])
        logger.info(loop_message)

    def before_backward_solve(self):
        if self.add_handlers and self.loop_index % self.handler_loop_cadence == 0:
            # setting tracer to end state of forward solve
            # self.forward_solver.state[1].change_scales(1)
            # self.backward_solver.state[1]['g'] = self.forward_solver.state[1]['g'].copy() - self.S['g'].copy()
            ex, ez = self.coords.unit_vector_fields(self.domain.dist)

            snapshots_backward = self.backward_solver.evaluator.add_file_handler(self.run_dir + '/' + self.write_suffix + '/snapshots_backward/snapshots_backward_loop' + str(self.loop_index), sim_dt=-0.01, max_writes=10, mode='overwrite')
            u_t = self.backward_solver.state[0]
            s_t = self.backward_solver.state[1]
            p_t = self.backward_solver.state[2]
            snapshots_backward.add_task(s_t, name='tracer')
            snapshots_backward.add_task(p_t, name='pressure')
            snapshots_backward.add_task(-d3.div(d3.skew(u_t)), name='vorticity')
            snapshots_backward.add_task(d3.dot(ex, u_t), name='ux')
            snapshots_backward.add_task(d3.dot(ez, u_t), name='uz')
        logger.debug('Starting backward solve')

    def during_backward_solve(self):
        # logger.debug('backward solver time = {}'.format(self.backward_solver.sim_time))
        pass

    def after_backward_solve(self):
        logger.debug('Completed backward solve')
        pass
