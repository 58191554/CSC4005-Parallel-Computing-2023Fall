#!/bin/bash
#SBATCH -o profiling/bonus.txt
#SBATCH -p Project
#SBATCH -J Project3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src
big_size=100000000
small_size=50
big_bucket=1000000
small_bucket=10

# Merge Sort
# OpenMP
echo "Merge Sort OpenMP (Optimized with -O2)"
for num_cores in  1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n $num_cores ${CURRENT_DIR}/../build/src/mergesort/mergesort_parallel ${num_cores} ${big_size}
done
echo ""

# Task 5
echo "Quick Sort OpenMP (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n $num_cores --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../build/src/quicksort/quicksort_parallel ${num_cores} ${big_size}
done
echo ""