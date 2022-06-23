#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dedalus3

MPIPROCS=1000
mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.2 0.05
mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.2 1.0
mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.2 3.0
mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.05 0.05
mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.05 1.0
mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.05 3.0