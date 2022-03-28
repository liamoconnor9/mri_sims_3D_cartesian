#PBS -S /bin/bash
#PBS -l select=19:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=18:00:00
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

SUFF="viff1en2_eta1div7500_R1p1547_N256_noslip1_Lx1e0_ARy3PIdiv8_ARz1PIdiv1_B0PPR_U01en2"
MPIPROC=512

mkdir $SUFF
cp $file $SUFF
cp mri_vp.py $SUFF

mpiexec_mpt -np $MPIPROC python3 mri_vp.py $SUFF
exit 1

cd $SUFF
mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs scalars --cleanup
mpiexec_mpt -np 1 python3 ../../plotting_scripts/plot_kebe.py scalars/*.h5 --suffix=$SUFF
mpiexec_mpt -np 1 python3 ../../plotting_scripts/plot_ke.py scalars/*.h5 --suffix=$SUFF
mpiexec_mpt -np 1 python3 ../../plotting_scripts/plot_be.py scalars/*.h5 --suffix=$SUFF
mv ../../plotting_scripts/*${SUFF}*.png .
mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs slicepoints --cleanup
mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_slicepoints_xy.py slicepoints/*.h5 --output=frames_xy --suffix=$SUFF
mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_slicepoints_xz.py slicepoints/*.h5 --output=frames_xz --suffix=$SUFF
mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_slicepoints_yz.py slicepoints/*.h5 --output=frames_yz --suffix=$SUFF
mpiexec_mpt -np $MPIPROC python3 ../../plotting_scripts/plot_kebe_profiles.py slicepoints/*.h5 --output=kebe_profiles --suffix=$SUFF
png2mp4 frames_xy/ mri_xy.mp4 60
png2mp4 frames_xz/ mri_xz.mp4 60
png2mp4 frames_yz/ mri_yz.mp4 60
png2mp4 kebe_profiles/ kebe_profiles.mp4 60
mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs checkpoints --cleanup