#PBS -S /bin/bash
#PBS -N mri_ivp_linear
#PBS -l select=5:ncpus=28:mpiprocs=140:model=bro
#PBS -l walltime=8:00:00
#PBS -j oe
module load mpi-sgi/mpt
module load comp-intel
export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true
cd ~/scratch/dedalus/mri/

mpiexec_mpt -np 128 python3 mri_ivp_linear.py
mpiexec_mpt -np 128 python3 -m dedalus merge_procs checkpoints_mri_wed
mpiexec_mpt -np 128 python3 -m dedalus merge_procs slicepoints_mri_wed
mpiexec_mpt -np 128 python3 plot_slices.py checkpoints_mri_wed/*.h5 --output=frames_wed