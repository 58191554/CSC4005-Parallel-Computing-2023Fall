#!/bin/bash
#SBATCH -o ./profiling/task5-cuda.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src
debug=1
# Define the perf command with required events
PERF_CMD="perf stat -e cache-references,cache-misses,page-faults,cycles,instructions -r 1"

echo "CUDA 4x4x4 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/cudaMul ${CURRENT_DIR}/../matrices/matrix1.txt ${CURRENT_DIR}/../matrices/matrix2.txt ${CURRENT_DIR}/../results/cuda/m12.txt $debug
echo ""

echo "CUDA 128x128x128 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/cudaMul ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../results/cuda/m34.txt $debug
echo ""

echo "CUDA 1024x1024 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/cudaMul ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/cuda/m56.txt $debug
echo ""

echo "CUDA 2048x2048 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/cudaMul ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/cuda/m78.txt $debug
echo ""
