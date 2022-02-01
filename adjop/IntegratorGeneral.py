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

class Integrator:
    def __init__(self, backward_problem, task_str, opt_params):
        self.backward_problem = backward_problem
        self.task_str = task_str
        self.opt_params = opt_params

