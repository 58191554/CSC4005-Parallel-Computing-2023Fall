#!/bin/bash
#SBATCH -o ./profiling/naive.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src
debug=0
# Define the perf command with required events
PERF_CMD="perf stat -e cache-references,cache-misses,page-faults,cycles,instructions -r 1"

# Naive
echo "Naive Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix1.txt ${CURRENT_DIR}/../matrices/matrix2.txt ${CURRENT_DIR}/../results/answers/m12.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../results/answers/m34.txt $debug
echo 
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/answers/m56.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/answers/m78.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/answers/m910.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/answers/m1112.txt $debug
echo 
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrixa.txt ${CURRENT_DIR}/../matrices/matrixb.txt ${CURRENT_DIR}/../results/answers/mab.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrixc.txt ${CURRENT_DIR}/../matrices/matrixd.txt ${CURRENT_DIR}/../results/answers/mcd.txt $debug
echo ""
