#PBS -S /bin/bash
#PBS -l select=2:ncpus=40:mpiprocs=40:model=sky_ele
#PBS -l walltime=12:00:00
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
conda activate dedalus-d3
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

source ~/png2mp4.sh
cd ~/scratch/dedalus/mri/adjop/shear

FILE="$(readlink -f "$0")"
DIR="$(dirname "$(readlink -f "$0")")/"
CONFIG="shear_options.cfg"

# If target simulation was previously run in OLDSUFFIX, just copy its contents over
SUFFIX="T3_N512_testcadence"

mkdir $SUFFIX/frames_error
mkdir $SUFFIX/movies_error

MPIPROC=80
mpiexec_mpt -np $MPIPROC python3 plot_snapshots_error.py $SUFFIX snapshots_target snapshots_forward frames_error
# mpiexec_mpt -np 1 python3 plot_errors.py $SUFFIX snapshots_target snapshots_forward frames_error
exit 1

for d in $SUFFIX/frames_error/*/ ; do
    MOVIE_NAME="$(basename $d)"
    png2mp4 $d $SUFFIX/movies_error/$MOVIE_NAME.mp4 60
done