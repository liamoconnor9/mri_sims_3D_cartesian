#PBS -S /bin/bash
#PBS -l select=1:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=8:00:00
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

# SUFFIX="T5_N256_reverse"
# OLDSUFFIX=$SUFFIX

python3 kdv_burgers.py
mpiexec_mpt -np 20 python3 kdv_parallel.py
mpiexec_mpt -np 1 python3 paths_pod.py
# python3 kdv_cg.py
# python3 kdv_angles.py
# python3 kdv_ts.py