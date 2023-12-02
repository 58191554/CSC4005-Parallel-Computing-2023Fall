#!/bin/bash
#SBATCH -o ./profiling/task4-mpi.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src
debug=0
# Define the perf command with required events
PERF_CMD="perf stat -e cache-references,cache-misses,page-faults,cycles,instructions -r 1"


# MPI + OpenMP + SIMD + Reordering
echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
echo "Matrix Multiply 1024x1024"
echo "Number of Processes: 1, Number of Threads: 32"
srun -n 1 --cpus-per-task 32 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
echo ""

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
echo ""

echo "Number of Processes: 4, Number of Threads: 8"
srun -n 4 --cpus-per-task 8 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
echo ""

echo "Number of Processes: 8, Number of Threads: 4"
srun -n 8 --cpus-per-task 4 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
echo ""

echo "Number of Processes: 16, Number of Threads: 2"
srun -n 16 --cpus-per-task 2 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
echo ""

echo "Number of Processes: 32, Number of Threads: 1"
srun -n 32 --cpus-per-task 1 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/MPI/m56.txt $debug
echo ""

echo "Matrix Multiply 2048x2048"
echo "Number of Processes: 1, Number of Threads: 32"
srun -n 1 --cpus-per-task 32 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
echo ""

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
echo ""

echo "Number of Processes: 4, Number of Threads: 8"
srun -n 4 --cpus-per-task 8 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
echo ""

echo "Number of Processes: 8, Number of Threads: 4"
srun -n 8 --cpus-per-task 4 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
echo ""

echo "Number of Processes: 16, Number of Threads: 2"
srun -n 16 --cpus-per-task 2 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
echo ""

echo "Number of Processes: 32, Number of Threads: 1"
srun -n 32 --cpus-per-task 1 --mpi=pmi2 $PERF_CMD ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/MPI/m78.txt $debug
echo ""
