#PBS -S /bin/bash
#PBS -N plot_points
#PBS -l select=19:ncpus=28:mpiprocs=512:model=bro
#PBS -l walltime=1:00:00
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
cd ~/scratch/dedalus/mri_simulations/

mpiexec_mpt -np 512 python3 -m dedalus merge_procs checkpoints_mri_non
mpiexec_mpt -np 512 python3 -m dedalus merge_procs slicepoints_mri_non
mpiexec_mpt -np 512 python3 plot_slices.py checkpoints_mri_wed/*.h5 --output=frames_non