#!/bin/bash

# Compile the src code in build
mkdir build && cd build
cmake ..
make -j4

# Run
sbatch src/sbach.sh