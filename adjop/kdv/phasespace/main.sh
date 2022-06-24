#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dedalus3

MPIPROCS=1372
# mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 5.0    0.03 
# mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.5    0.3
# mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.05   3.0

# mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 3.0    0.05 
mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 1.0    0.15 
# mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.15   1.0

# mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.05 0.05
# mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.05 1.0
# mpiexec_mpt -np $MPIPROCS python3 kdv_twomodes.py 0.05 3.0