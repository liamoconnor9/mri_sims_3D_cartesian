#!/bin/bash
for d in */ ; do
    echo "Directory $d contains"
    find $d -type f | wc -l
done