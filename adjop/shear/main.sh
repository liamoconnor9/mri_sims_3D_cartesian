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

# If target simulation was previously run in OLDSUFFIX, just copy its contents over
SUFFIX="T5_N256_test"
OLDSUFFIX=$SUFFIX
# OLDSUFFIX="T1_0ic"
MPIPROC=32

mkdir $SUFFIX
mkdir $SUFFIX/snapshots_forward
mkdir $SUFFIX/snapshots_backward
mkdir $SUFFIX/frames_forward
mkdir $SUFFIX/frames_backward
mkdir $SUFFIX/frames_target
mkdir $SUFFIX/movies_forward
mkdir $SUFFIX/movies_backward

if [ -v OLDSUFFIX ];then
    cp -r $OLDSUFFIX/checkpoint_target/ $SUFFIX/checkpoint_target/
    cp $OLDSUFFIX/movie_target.mp4 $SUFFIX/movie_target.mp4
else
    mpiexec_mpt -np $MPIPROC python3 shear_flow.py $CONFIG $SUFFIX
    mpiexec_mpt -np $MPIPROC python3 plot_snapshots_og.py $SUFFIX snapshots_target frames_target
    png2mp4 $SUFFIX/frames_target/ $SUFFIX/movie_target.mp4 60
fi

mpiexec_mpt -np $MPIPROC python3 shear_cg.py $CONFIG $SUFFIX
exit 1
MPIPROC=80
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

# Do you think it's possible that adjoint looping only works well for low-diffusivity problems? Like, for example, if you wanted to find the initial condition associated with a given end state for the diffusion equation, I think you'd be SOL because the backward problem would also be the diffusion equation. But for reversible problems like kdv and the wave equation, the gradient is as accurate as the simulations