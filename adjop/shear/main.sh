#PBS -S /bin/bash
#PBS -l select=2:ncpus=40:mpiprocs=40:model=sky_ele
#PBS -l walltime=2:00:00
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
SUFFIX="T1p5"
# SUFFIX="T3_N512_LOOP30_INTER"
MPIPROC=64
mkdir $SUFFIX
mkdir $SUFFIX/snapshots
mkdir $SUFFIX/frames
mkdir $SUFFIX/movies

# mpiexec_mpt -np $MPIPROC python3 shear_flow.py
mpiexec_mpt -np $MPIPROC python3 main_shear.py $SUFFIX
mpiexec_mpt -np $MPIPROC python3 plot_snapshots.py $SUFFIX

for d in $SUFFIX/frames/*/ ; do
    MOVIE_NAME="$(basename $d)"
    png2mp4 $d $SUFFIX/movies/$MOVIE_NAME.mp4 60
done