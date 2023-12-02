#!/bin/bash
#SBATCH -o ./profiling/task1-memory-locality.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src
debug=0
# Define the perf command with required events
PERF_CMD="perf stat -e cache-references,cache-misses,page-faults,cycles,instructions -r 1"

# Memory Locality
echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix1.txt ${CURRENT_DIR}/../matrices/matrix2.txt ${CURRENT_DIR}/../results/mem_local/m12.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../results/mem_local/m34.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/mem_local/m56.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mem_local/m78.txt $debug
echo ""
