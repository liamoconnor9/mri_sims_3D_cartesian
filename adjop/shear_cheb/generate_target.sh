#PBS -S /bin/bash
#PBS -l select=2:ncpus=40:mpiprocs=40:model=sky_ele
#PBS -l walltime=4:00:00
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
# SUFFIX="T3_N512_vorticity"
SUFFIX="T0p5_N512_coeff0p0_Re2e4_oldic"
# OLDSUFFIX="T3_N256_coeff0p90_Re2e4"
# OLDSUFFIX=$SUFFIX
MPIPROC=64

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

mpiexec_mpt -np $MPIPROC python3 shear_flow.py $CONFIG $SUFFIX
mpiexec_mpt -np $MPIPROC python3 plot_snapshots_og.py $SUFFIX snapshots_target frames_target
png2mp4 $SUFFIX/frames_target/ $SUFFIX/movie_target.mp4 60