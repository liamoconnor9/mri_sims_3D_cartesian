#!/bin/bash
echo $"###################################################"
mpiexec_mpt -np 1 python3 vp_lsa.py mri_options.cfg 0.0 0.25
mpiexec_mpt -np 1 python3 vp_lsa.py mri_options.cfg 0.0 0.5
mpiexec_mpt -np 1 python3 vp_lsa.py mri_options.cfg 0.25 0.0
mpiexec_mpt -np 1 python3 vp_lsa.py mri_options.cfg 0.25 0.25
mpiexec_mpt -np 1 python3 vp_lsa.py mri_options.cfg 0.25 0.5
mpiexec_mpt -np 1 python3 vp_lsa.py mri_options.cfg 0.5 0.0
mpiexec_mpt -np 1 python3 vp_lsa.py mri_options.cfg 0.5 0.25
mpiexec_mpt -np 1 python3 vp_lsa.py mri_options.cfg 0.5 0.5