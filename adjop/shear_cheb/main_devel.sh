#PBS -S /bin/bash
#PBS -l select=3:ncpus=28:mpiprocs=28:model=bro
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

FILE="$(readlink -f "$0")"
DIR="$(dirname "$(readlink -f "$0")")/"
CONFIG="shear_options_devel.cfg"

cd ~/mri/adjop/shear_cheb
# SUFFIX="T1_coeff0ic_negbic_re1e4_N256"

# If target simulation was previously run in OLDSUFFIX, just copy its contents over
SUFFIX="T3_N512_re1e5"
MPIPROC=64
# OLDSUFFIX=$SUFFIX

if [ ! -d "$SUFFIX" ]; then

    mkdir $SUFFIX
    mkdir $SUFFIX/checkpoints
    mkdir $SUFFIX/snapshots_forward
    mkdir $SUFFIX/snapshots_backward
    mkdir $SUFFIX/frames_forward
    mkdir $SUFFIX/frames_backward
    mkdir $SUFFIX/frames_target
    mkdir $SUFFIX/frames_error
    mkdir $SUFFIX/movies_forward
    mkdir $SUFFIX/movies_backward
    mkdir $SUFFIX/movies_error

    if [ -v OLDSUFFIX ];then
        cp -r $OLDSUFFIX/checkpoint_target/ $SUFFIX/checkpoint_target/
        cp -r $OLDSUFFIX/snapshots_target/ $SUFFIX/snapshots_target/
        cp $OLDSUFFIX/movie_target.mp4 $SUFFIX/movie_target.mp4
    else
        mpiexec_mpt -np $MPIPROC python3 shear_flow.py $CONFIG $SUFFIX
        mpiexec_mpt -np $MPIPROC python3 plot_snapshots_og.py $SUFFIX snapshots_target frames_target
        png2mp4 $SUFFIX/frames_target/ $SUFFIX/movie_target.mp4 60
    fi

fi

mpiexec_mpt -np $MPIPROC python3 shear_cg.py $CONFIG $SUFFIX
exit 1
# MPIPROC=10
# mpiexec_mpt -np $MPIPROC python3 plot_snapshots_error.py $SUFFIX snapshots_target snapshots_forward frames_error
# mpiexec_mpt -np 1 python3 plot_errors.py $SUFFIX snapshots_target snapshots_forward frames_error


# MPIPROC=40
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
MPIPROC=10
mpiexec_mpt -np $MPIPROC python3 plot_snapshots_error.py $SUFFIX snapshots_target snapshots_forward frames_error
mpiexec_mpt -np 1 python3 plot_errors.py $SUFFIX snapshots_target snapshots_forward frames_error
exit 1
for d in $SUFFIX/frames_error/*/ ; do
    MOVIE_NAME="$(basename $d)"
    png2mp4 $d $SUFFIX/movies_error/$MOVIE_NAME.mp4 60
done