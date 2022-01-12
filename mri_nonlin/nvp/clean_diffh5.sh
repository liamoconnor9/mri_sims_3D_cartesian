#!/bin/bash
for d in diff*/ ; do
    echo "$d"
    rm -rf $d/scalars*/*/
    rm -rf $d/slicepoints*/*/
    rm -rf $d/checkpoints*/*/
    # find $d -type f | wc -l
done