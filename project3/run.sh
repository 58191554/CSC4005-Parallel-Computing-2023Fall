#!/bin/bash

# Compile the src code in build
mkdir build 
cd build
cmake ..
make -j4

# Run
cd ..
sbatch src/sbach.sh
sbatch src/sbach_bonus.sh