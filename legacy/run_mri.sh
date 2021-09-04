#!/usr/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=faculty

#conda activate dedalus

date
mpiexec -np 28 python3 mri.py runs/run_1.cfg
date
