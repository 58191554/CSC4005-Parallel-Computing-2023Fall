#!/bin/bash
#SBATCH -o ./Project2-Results.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src
debug=1
# Define the perf command with required events
# PERF_CMD="perf stat -e cache-references,cache-misses,page-faults,cycles,instructions -r 1"
PERF_CMD=""

# Naive
# echo "Naive Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix1.txt ${CURRENT_DIR}/../matrices/matrix2.txt ${CURRENT_DIR}/../results/naives/naive12.txt $debug
# echo ""
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../results/naives/naive34.txt $debug
# echo 
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/naives/naive56.txt 0
# echo ""
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/naives/naive78.txt $debug
# echo ""

# Memory Locality
# echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix1.txt ${CURRENT_DIR}/../matrices/matrix2.txt ${CURRENT_DIR}/../results/mem_local/m12.txt $debug
# echo ""
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../results/mem_local/m34.txt $debug
# echo ""
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/mem_local/m56.txt 1
# echo ""
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mem_local/m78.txt $debug
# echo ""

# SIMD + Reordering
# echo "SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix1.txt ${CURRENT_DIR}/../matrices/matrix2.txt ${CURRENT_DIR}/../results/simd/m12.txt $debug
# echo ""
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../results/simd/m34.txt $debug
# echo ""
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/simd/m56.txt 1
# echo ""
# srun -n 1 --cpus-per-task 1 $PERF_CMD ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/simd/m78.txt $debug
# echo ""

# # OpenMP + SIMD + Reordering
# echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"

# for num_cores in 1 2 4 8 16 32
# do
#   echo "4x4 Number of cores: $num_cores"
#   srun -n 1 --cpus-per-task $num_cores $PERF_CMD ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix1.txt ${CURRENT_DIR}/../matrices/matrix2.txt ${CURRENT_DIR}/../results/openMP/m12.txt $debug
#   echo ""
# done

# for num_cores in 1 2 4 8 16 32
# do
#   echo "128x128 Number of cores: $num_cores"
#   srun -n 1 --cpus-per-task $num_cores $PERF_CMD ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../results/openMP/m34.txt $debug
#   echo ""
# done


# for num_cores in  1 2 4 8 16 32
# do
#   echo "1024x1024 Number of cores: $num_cores"
#   srun -n 1 --cpus-per-task $num_cores $PERF_CMD ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/openMP/m56.txt 1
#   echo ""
# done

# for num_cores in  1 2 4 8 16 32
# do
#   echo "2048x2048 Number of cores: $num_cores"
#   srun -n 1 --cpus-per-task $num_cores $PERF_CMD ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/openMP/m78.txt 1
#   echo ""
# done

# MPI + OpenMP + SIMD + Reordering
# echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# echo "Matrix Multiply 1024x1024"
# echo "Number of Processes: 1, Number of Threads: 32"
# srun -n 1 --cpus-per-task 32 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
# echo ""

# echo "Number of Processes: 2, Number of Threads: 16"
# srun -n 2 --cpus-per-task 16 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
# echo ""

# echo "Number of Processes: 4, Number of Threads: 8"
# srun -n 4 --cpus-per-task 8 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
# echo ""

# echo "Number of Processes: 8, Number of Threads: 4"
# srun -n 8 --cpus-per-task 4 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
# echo ""

# echo "Number of Processes: 16, Number of Threads: 2"
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
# echo ""

# echo "Number of Processes: 32, Number of Threads: 1"
# srun -n 32 --cpus-per-task 1 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
# echo ""

# echo "Matrix Multiply 2048x2048"
# echo "Number of Processes: 1, Number of Threads: 32"
# srun -n 1 --cpus-per-task 32 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
# echo ""

# echo "Number of Processes: 2, Number of Threads: 16"
# srun -n 2 --cpus-per-task 16 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
# echo ""

# echo "Number of Processes: 4, Number of Threads: 8"
# srun -n 4 --cpus-per-task 8 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
# echo ""

# echo "Number of Processes: 8, Number of Threads: 4"
# srun -n 8 --cpus-per-task 4 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
# echo ""

# echo "Number of Processes: 16, Number of Threads: 2"
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
# echo ""

# echo "Number of Processes: 32, Number of Threads: 1"
# srun -n 32 --cpus-per-task 1 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
# echo ""


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

# echo "OpenACC 4x4x4 Matrix Mulitiplication"
# srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/openaccMul ${CURRENT_DIR}/../matrices/matrix1.txt ${CURRENT_DIR}/../matrices/matrix2.txt ${CURRENT_DIR}/../results/openacc/m12.txt $debug
# echo ""

# echo "OpenACC 128x128x128 Matrix Mulitiplication"
# srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/openaccMul ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../results/openacc/m34.txt $debug
# echo ""

# echo "OpenACC 1024x1024 Matrix Mulitiplication"
# srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/openaccMul ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/openacc/m56.txt $debug
# echo ""

# echo "OpenACC 2048x2048 Matrix Mulitiplication"
# srun -n 1 --gpus 1 $PERF_CMD ${CURRENT_DIR}/../build/src/gpu/openaccMul ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/openacc/m78.txt $debug
# echo ""
