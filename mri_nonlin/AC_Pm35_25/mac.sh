#PBS -S /bin/bash
#PBS -l select=19:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -W group_list=s2276
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

# module load mpi-sgi/mpt
# module load comp-intel
export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

source ~/png2mp4.sh
cd ~/scratch/dedalus/mri/mri_nonlin

SUFF="AC_Pm35_20"

FILE="$(readlink -f "$0")"
DIR="$(dirname "$(readlink -f "$0")")/"
CONFIG="mri_options.cfg"

MPIPROC=512

mkdir $SUFF
cp $FILE $SUFF
cp $CONFIG $SUFF
cp mri_ac.py $SUFF

# mpiexec_mpt -np $MPIPROC python3 mri_ac.py $CONFIG $DIR $SUFF
cd $SUFF

mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs scalars --cleanup
mpiexec_mpt -np 1 python3 ../../plotting_scripts/plot_kebe.py scalars/*.h5 --dir=$DIR --config=$CONFIG --suffix=$SUFF
mpiexec_mpt -np 1 python3 ../../plotting_scripts/plot_ke.py scalars/*.h5 --dir=$DIR --config=$CONFIG --suffix=$SUFF
mpiexec_mpt -np 1 python3 ../../plotting_scripts/plot_be.py scalars/*.h5 --dir=$DIR --config=$CONFIG --suffix=$SUFF
mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs slicepoints --cleanup
mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_slicepoints_xy.py slicepoints/*.h5 --output=frames_xy --dir=$DIR --config=$CONFIG --suffix=$SUFF
mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_slicepoints_xz.py slicepoints/*.h5 --output=frames_xz --dir=$DIR --config=$CONFIG --suffix=$SUFF
mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_slicepoints_yz.py slicepoints/*.h5 --output=frames_yz --dir=$DIR --config=$CONFIG --suffix=$SUFF
mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_kebe_profiles.py slicepoints/*.h5 --output=kebe_profiles --dir=$DIR --config=$CONFIG --suffix=$SUFF
png2mp4 frames_xy/ mri_xy.mp4 60
png2mp4 frames_xz/ mri_xz.mp4 60
png2mp4 frames_yz/ mri_yz.mp4 60
png2mp4 kebe_profiles/ kebe_profiles.mp4 60
mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs checkpoints --cleanup