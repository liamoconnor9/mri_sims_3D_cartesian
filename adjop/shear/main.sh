#PBS -S /bin/bash
#PBS -l select=1:ncpus=40:mpiprocs=40:model=sky_ele
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
CONFIG="shear_options.cfg"
# SUFFIX="T1_coeff0ic_negbic_re1e4_N256"
SUFFIX="temp"
MPIPROC=32

mkdir $SUFFIX
mkdir $SUFFIX/snapshots_forward
mkdir $SUFFIX/snapshots_backward
mkdir $SUFFIX/frames_forward
mkdir $SUFFIX/frames_backward
mkdir $SUFFIX/frames_target
mkdir $SUFFIX/movies_forward
mkdir $SUFFIX/movies_backward

mpiexec_mpt -np $MPIPROC python3 shear_flow.py $CONFIG $SUFFIX
mpiexec_mpt -np $MPIPROC python3 plot_snapshots_og.py $SUFFIX snapshots_target frames_target
png2mp4 $SUFFIX/frames_target/ $SUFFIX/movie_target.mp4 60

mpiexec_mpt -np $MPIPROC python3 shear_cg.py $CONFIG $SUFFIX
mpiexec_mpt -np $MPIPROC python3 plot_snapshots.py $SUFFIX snapshots_forward frames_forward
mpiexec_mpt -np $MPIPROC python3 plot_snapshots.py $SUFFIX snapshots_backward frames_backward

for d in $SUFFIX/frames_forward/*/ ; do
    MOVIE_NAME="$(basename $d)"
    png2mp4 $d $SUFFIX/movies_forward/$MOVIE_NAME.mp4 60
done
for d in $SUFFIX/frames_backward/*/ ; do
    MOVIE_NAME="$(basename $d)"
    png2mp4 $d $SUFFIX/movies_backward/$MOVIE_NAME.mp4 60
done
exit 1