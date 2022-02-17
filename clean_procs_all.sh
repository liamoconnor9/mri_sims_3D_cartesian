#!/bin/bash

for i in {0..1024}
    do
        find . -name "*p${i}.h5" -delete
    done