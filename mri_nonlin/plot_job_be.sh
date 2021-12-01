#PBS -S /bin/bash
#PBS -l select=1:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=8:00:00
#PBS -j oe
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

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

cd ~/scratch/dedalus/mri/mri_nonlin

SUFF="diff1en2_R1p01_lr"

mpiexec_mpt -np 1 python3 plot_be_dirs.py ${SUFF}/checkpoints_${SUFF}/*.h5 --suffix=$SUFF