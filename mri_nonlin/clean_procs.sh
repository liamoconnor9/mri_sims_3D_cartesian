#!/bin/bash

TYPE=$1

for dir in ${TYPE}_*/   # list directories in the form "/tmp/dirname/"
do
    echo "Cleaning procs from ${dir}"
    rm -rf $dir/*/
done
