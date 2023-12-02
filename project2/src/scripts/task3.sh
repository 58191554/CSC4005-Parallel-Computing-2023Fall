#!/bin/bash
#SBATCH -o ./profiling/task3-OpenMP-SIMD-Reordering.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src
debug=0
# Define the perf command with required events
PERF_CMD="perf stat -e cache-references,cache-misses,page-faults,cycles,instructions -r 1"


# OpenMP + SIMD + Reordering
echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"

for num_cores in 1 2 4 8 16 32
do
  echo "4x4 Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores $PERF_CMD ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix1.txt ${CURRENT_DIR}/../matrices/matrix2.txt ${CURRENT_DIR}/../results/openMP/m12.txt $debug
  echo ""
done

for num_cores in 1 2 4 8 16 32
do
  echo "128x128 Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores $PERF_CMD ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../results/openMP/m34.txt $debug
  echo ""
done


for num_cores in  1 2 4 8 16 32
do
  echo "1024x1024 Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores $PERF_CMD ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/openMP/m56.txt $debug
  echo ""
done

for num_cores in  1 2 4 8 16 32
do
  echo "2048x2048 Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores $PERF_CMD ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/openMP/m78.txt $debug
  echo ""
done
