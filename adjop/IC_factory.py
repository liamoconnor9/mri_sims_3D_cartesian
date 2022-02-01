from unicodedata import decimal
from docopt import docopt
import time
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import os
import sys
import h5py
import gc
import pickle
import dedalus.public as de
from dedalus.core.field import Scalar
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import plot_bot_2d
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)

class IC_factory:
    ##### Initial condition functions from Evan H. Anders
    def filter_field(field, frac=0.25):
        """
        Filter a field in coefficient space by cutting off all coefficient above
        a given threshold.  This is accomplished by changing the scale of a field,
        forcing it into coefficient space at that small scale, then coming back to
        the original scale.
        Inputs:
            field   - The dedalus field to filter
            frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                        The upper 75% of coefficients are set to 0.
        """

        logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
        orig_scale = field.scales
        field.set_scales(frac, keep_data=True)
        field['c']
        field['g']
        field.set_scales(orig_scale, keep_data=True)
        
    def global_noise(domain, magnitude, seed=42, **kwargs):
        """
        Create a field filled with random noise of order 1.  Modify seed to
        get varying noise, keep seed the same to directly compare runs.
        """
        # Random perturbations, initialized globally for same results in parallel
        gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
        slices = domain.dist.grid_layout.slices(scales=domain.dealias)
        rand = np.random.RandomState(seed=seed)
        noise = rand.standard_normal(gshape)[slices]
        # filter in k-space
        noise_field = domain.new_field()
        noise_field.set_scales(domain.dealias, keep_data=False)
        noise_field['g'] = noise * magnitude
        IC_factory.filter_field(noise_field, **kwargs)
        return noise_field

    def build_ic(domain, magnitude):
        return IC_factory.global_noise(domain, magnitude)
