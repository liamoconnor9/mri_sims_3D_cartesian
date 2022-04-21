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
SUFFIX="temp"
MPIPROC=64

mkdir $SUFFIX
mkdir $SUFFIX/snapshots
mkdir $SUFFIX/snapshots_backward
mkdir $SUFFIX/frames
mkdir $SUFFIX/frames_backward
mkdir $SUFFIX/movies

# mpiexec_mpt -np $MPIPROC python3 shear_flow.py
mpiexec_mpt -np $MPIPROC python3 main_shear.py $SUFFIX
exit 1
mpiexec_mpt -np $MPIPROC python3 plot_snapshots.py $SUFFIX snapshots frames
mpiexec_mpt -np $MPIPROC python3 plot_snapshots.py $SUFFIX snapshots_backward frames_backward

for d in $SUFFIX/frames/*/ ; do
    MOVIE_NAME="$(basename $d)"
    png2mp4 $d $SUFFIX/movies/$MOVIE_NAME.mp4 60
done
for d in $SUFFIX/frames_backward/*/ ; do
    MOVIE_NAME="$(basename $d)"
    png2mp4 $d $SUFFIX/movies/$MOVIE_NAME.mp4 60
done