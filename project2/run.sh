#!/bin/bash

# create a dir to store results
if [ ! -d "results" ]; then
    # If it doesn't exist, create the 'results' directory and change to it
    mkdir -p results
    echo "[Build Directory for results]"
fi

if [ ! -d "results/answers" ]; then
    mkdir -p results/answers
    echo "[Build Directory for answers]"
fi

if [ ! -d "results/naives" ]; then
    mkdir -p results/naives
    echo "[Build Directory for naives]"
fi

if [ ! -d "results/openMP" ]; then
    mkdir -p results/openMP
    echo "[Build Directory for openMP]"
fi

if [ ! -d "results/MPI" ]; then
    mkdir -p results/MPI
    echo "[Build Directory for MPI]"
fi

if [ ! -d "results/cuda" ]; then
    mkdir -p results/cuda
    echo "[Build Directory for cuda]"
fi

# Create the 'build' directory if it doesn't exist and change to it
mkdir -p build
cd build

# Run CMake to configure the project
# Use -DCMAKE_BUILD_TYPE=Debug for debug build error message logging
# Use the appropriate 'cmake' or 'cmake3' command based on your environment
cmake ..

# Build the project using 'make' with 4 parallel jobs
make -j4

# execute sbatch.sh
cd ..
sbatch src/sbach.sh 