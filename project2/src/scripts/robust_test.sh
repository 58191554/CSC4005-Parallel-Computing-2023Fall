#!/bin/bash
#SBATCH -o ./Project2-Robust-Results.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src
debug=1

Naive
echo "Naive Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/answers/m910.txt 0
echo ""
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/answers/m1112.txt 0
echo ""
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrixa.txt ${CURRENT_DIR}/../matrices/matrixb.txt ${CURRENT_DIR}/../results/answers/mab.txt 0
echo ""
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrixc.txt ${CURRENT_DIR}/../matrices/matrixd.txt ${CURRENT_DIR}/../results/answers/mcd.txt 0
echo ""


# Memory Locality
echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/answers/m910.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/answers/m1112.txt $debug
echo ""

SIMD + Reordering
echo "SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/simd/m910.txt $debug
echo ""
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/simd/m1112.txt $debug
echo ""

# OpenMP + SIMD + Reordering
echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"

for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/openMP/m910.txt $debug
  echo ""
done

for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/openMP/m1112.txt $debug
  echo ""
done


# MPI + OpenMP + SIMD + Reordering
echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
echo "Matrix Multiply 127x126x125"
echo "Number of Processes: 1, Number of Threads: 32"
srun -n 1 --cpus-per-task 32 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/MPI/m910.txt $debug
echo ""

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/MPI/m910.txt $debug
echo ""

echo "Number of Processes: 4, Number of Threads: 8"
srun -n 4 --cpus-per-task 8 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/MPI/m910.txt $debug
echo ""

echo "Number of Processes: 8, Number of Threads: 4"
srun -n 8 --cpus-per-task 4 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/MPI/m910.txt $debug
echo ""

echo "Number of Processes: 16, Number of Threads: 2"
srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/MPI/m910.txt $debug
echo ""

echo "Number of Processes: 32, Number of Threads: 1"
srun -n 32 --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/MPI/m910.txt $debug
echo ""

echo "Number of Processes: 1, Number of Threads: 32"
srun -n 1 --cpus-per-task 32 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/MPI/m1112.txt $debug
echo ""

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/MPI/m1112.txt $debug
echo ""

echo "Number of Processes: 4, Number of Threads: 8"
srun -n 4 --cpus-per-task 8 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/MPI/m1112.txt $debug
echo ""

echo "Number of Processes: 8, Number of Threads: 4"
srun -n 8 --cpus-per-task 4 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/MPI/m1112.txt $debug
echo ""

echo "Number of Processes: 16, Number of Threads: 2"
srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/MPI/m1112.txt $debug
echo ""

echo "Number of Processes: 32, Number of Threads: 1"
srun -n 32 --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/MPI/m1112.txt $debug
echo ""

echo "CUDA 4x4x4 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/cudaMul ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/cuda/m910.txt $debug
echo ""

echo "CUDA 128x128x128 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/cudaMul ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/cuda/m1112.txt $debug
echo ""

echo "CUDA 3x7x5 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/cudaMul ${CURRENT_DIR}/../matrices/matrixa.txt ${CURRENT_DIR}/../matrices/matrixb.txt ${CURRENT_DIR}/../results/cuda/mab.txt $debug
echo ""

echo "OpenACC 4x4x4 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/openaccMul ${CURRENT_DIR}/../matrices/matrix9.txt ${CURRENT_DIR}/../matrices/matrix10.txt ${CURRENT_DIR}/../results/openacc/m910.txt $debug
echo ""

echo "OpenACC 128x128x128 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/openaccMul ${CURRENT_DIR}/../matrices/matrix11.txt ${CURRENT_DIR}/../matrices/matrix12.txt ${CURRENT_DIR}/../results/openacc/m1112.txt $debug
echo ""

echo "OpenACC 1024x1024 Matrix Mulitiplication"
srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/openaccMul ${CURRENT_DIR}/../matrices/matrixa.txt ${CURRENT_DIR}/../matrices/matrixb.txt ${CURRENT_DIR}/../results/openacc/mab.txt $debug
echo ""

