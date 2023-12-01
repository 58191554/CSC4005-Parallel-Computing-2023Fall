#!/bin/bash

# Create build directory
mkdir -p build

# Navigate to the build directory
cd build

# Run cmake
cmake ..

# Run make
make

# Navigate back to the original directory
cd ..
chmod +x run_pc.sh
./run_pc.sh