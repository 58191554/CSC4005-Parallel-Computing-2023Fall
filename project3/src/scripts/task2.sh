#!/bin/bash
#SBATCH -o profiling/task2.txt
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
perf_command="perf stat -e cycles,instructions,cache-references,cache-misses"

# Bucket Sort
# Sequential
# echo "Bucket Sort Sequential (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/bucketsort/bucketsort_sequential 100000000 1000000
# echo ""
# MPI
echo "Bucket Sort MPI (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n $num_cores --cpus-per-task 1 --mpi=pmi2 $perf_command ${CURRENT_DIR}/../build/src/bucketsort/bucketsort_mpi ${big_size} ${big_bucket}
done
echo ""

