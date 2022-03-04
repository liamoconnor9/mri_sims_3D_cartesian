#PBS -S /bin/bash
#PBS -l select=10:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -W group_list=s2276
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

shopt -s expand_aliases
alias mpiexec_mpt="mpirun"
alias ffmpeg3="ffmpeg"

export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus-d3
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

source ~/png2mp4.sh
cd ~/mri/mri_d3

SUFF="jiff1en2_R0p6_Bsin2x_N32_noslip1_T1e2_test3"
MPIPROC=4

mkdir $SUFF
cp $file $SUFF
cp mri_adj.py $SUFF
cd $SUFF

mpiexec_mpt -np $MPIPROC python3 mri_adj.py $SUFF
# # mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs scalars_${SUFF} --cleanup
# mpiexec_mpt -np 1 python3 ../../plotting_scripts/plot_kebe.py scalars_${SUFF}/*.h5 --suffix=$SUFF
# mpiexec_mpt -np 1 python3 ../../plotting_scripts/plot_ke.py scalars_${SUFF}/*.h5 --suffix=$SUFF
# mpiexec_mpt -np 1 python3 ../../plotting_scripts/plot_be.py scalars_${SUFF}/*.h5 --suffix=$SUFF
# mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs slicepoints_viff1en2_R0p6_Bsin2x_N16 --cleanup
# mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_slicepoints_xy.py slicepoints_${SUFF}/*.h5 --output=frames_xy_${SUFF}
# mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_slicepoints_xz.py slicepoints_${SUFF}/*.h5 --output=frames_xz_${SUFF}
# mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_slicepoints_yz.py slicepoints_${SUFF}/*.h5 --output=frames_yz_${SUFF}
# mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_kebe_profiles.py slicepoints_${SUFF}/*.h5 --output=kebe_profiles_${SUFF}
# png2mp4 frames_xy_${SUFF}/ mri_${SUFF}_xy.mp4 60
# png2mp4 frames_xz_${SUFF}/ mri_${SUFF}_xz.mp4 60
# png2mp4 frames_yz_${SUFF}/ mri_${SUFF}_yz.mp4 60
# png2mp4 kebe_profiles_${SUFF}/ kebe_profiles_${SUFF}.mp4 60
# # mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs checkpoints_${SUFF} --cleanup
