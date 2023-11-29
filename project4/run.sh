#!/bin/bash

# Create build directory
mkdir -p build

# Navigate to the build directory
cd build

# Run cmake
cmake ..

# Run make
make

cd ..

chmod +x sbatch.sh
chmod +x test.sh
sbatch ./sbatch.sh