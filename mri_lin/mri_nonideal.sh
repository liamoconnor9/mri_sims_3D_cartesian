#PBS -S /bin/bash
#PBS -N mri_nonideal_diff2en4
#PBS -l select=1:ncpus=40:mpiprocs=40:model=sky_ele
#PBS -l walltime=16:00:00
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
source png2mp4.sh

mpiexec_mpt -np 32 python3 mri_nonideal_bc.py
mpiexec_mpt -np 32 python3 -m dedalus merge_procs slicepoints_diff2en4
mpiexec_mpt -np 32 python3 plot_slicepoints.py slicepoints_diff2en4/*.h5 --output=frames_diff2en4
png2mp4 frames_diff2en4/ mri_diff2en4.mp4 60
mpiexec_mpt -np 1 python3 plot_ke.py slicepoints_diff2en4/*.h5
mpiexec_mpt -np 1 python3 plot_be.py slicepoints_diff2en4/*.h5
mpiexec_mpt -np 32 python3 -m dedalus merge_procs checkpoints_diff2en4