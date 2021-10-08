#PBS -S /bin/bash
#PBS -N mri_nonlin_diff1en3_L8
#PBS -l select=10:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=24:00:00
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

source ~/png2mp4.sh
cd ~/scratch/dedalus/mri/mri_nonlin

SUFF="diff1en3_L8_asym"
DIFF=0.001
MPIPROC=256

mpiexec_mpt -np $MPIPROC python3 mri_nonlin.py $SUFF $DIFF
mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs scalars_${SUFF}
mpiexec_mpt -np 1 python3 plot_ke.py scalars_${SUFF}/*.h5 --suffix=$SUFF
mpiexec_mpt -np 1 python3 plot_be.py scalars_${SUFF}/*.h5 --suffix=$SUFF
mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs slicepoints_${SUFF}
mpiexec_mpt -np $MPIPROC python3 plot_slicepoints.py slicepoints_${SUFF}/*.h5 --output=frames_${SUFF}
png2mp4 frames_${SUFF}/ mri_${SUFF}.mp4 60
mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs checkpoints_${SUFF}

