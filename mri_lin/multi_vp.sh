#!/bin/bash
echo $"###################################################"
mpiexec_mpt -np 1 python3 vp_lsa.py mri_options.cfg $1