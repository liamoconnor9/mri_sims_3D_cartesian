#PBS -S /bin/bash
#PBS -l select=5:ncpus=40:mpiprocs=40:model=sky_ele
#PBS -l walltime=20:00:00
#PBS -j oe
#PBS -W group_list=s2276
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus3
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

source ~/png2mp4.sh
cd ~/scratch/dedalus/mri/adjop/kdv

FILE="$(readlink -f "$0")"
DIR="$(dirname "$(readlink -f "$0")")/"
CONFIG="kdv_options.cfg"
# CONFIG="kdv_sines.cfg"
PROCS=200

mpiexec_mpt -np 1      python3 kdv_burgers.py         $CONFIG
mpiexec_mpt -np $PROCS python3 kdv_burgers_sphere.py  $CONFIG
mpiexec_mpt -np $PROCS python3 kdv_parallel.py        $CONFIG
mpiexec_mpt -np 1      python3 paths_pod.py           $CONFIG