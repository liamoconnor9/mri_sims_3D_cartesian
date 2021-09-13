#!/bin/bash
SUFF1="diff1en2_R1p01"
SUFF2="diff1en2_R1p01_test"

for i in {1..75}
    do
        cp "checkpoints_${SUFF1}/checkpoints_${SUFF1}_s${i}.h5" "checkpoints_${SUFF2}/"
    done