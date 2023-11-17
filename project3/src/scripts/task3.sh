#!/bin/bash
#SBATCH -o profiling/task3.txt
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

oddeven_size=200000

# Odd-Even Sort
# Sequential
# echo "Odd-Even Sort Sequential (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/odd-even-sort/odd-even-sort_sequential ${oddeven_size}
# echo ""
# MPI
echo "Odd-Even Sort MPI (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n $num_cores --cpus-per-task 1 --mpi=pmi2 $perf_command ${CURRENT_DIR}/../build/src/odd-even-sort/odd-even-sort_mpi ${oddeven_size}
done
echo ""

