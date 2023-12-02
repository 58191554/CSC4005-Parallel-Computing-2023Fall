#!/bin/bash

# Change directory to 'build/'
cd build/

# Run 'make' with 4 parallel jobs
make -j4

# Change back to the parent directory
cd ..

# Submit the job to SLURM using 'sbatch'
sbatch src/scripts/sbatch_PartB.sh
