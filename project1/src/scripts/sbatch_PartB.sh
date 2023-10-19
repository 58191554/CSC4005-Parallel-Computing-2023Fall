#!/bin/bash
#SBATCH -o ./Project1-PartB-Results.txt
#SBATCH -p Project
#SBATCH -J Project1-PartB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Get the current directory
CURRENT_DIR=$(pwd)/src/scripts
echo "Current directory: ${CURRENT_DIR}"

# Sequential PartB
echo "Sequential PartB (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/sequential_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth-sequential.jpg
echo ""

# SIMD PartB
echo "SIMD(AVX2) PartB (Optimized with -O2)"
mkdir images/smooth-SIMD
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/simd_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/smooth-SIMD/20K-Smooth-SIMD.jpg
echo ""

# MPI PartB
mkdir images/smooth-MPI
for num_processes in 1 2 4 8 16 32
do
  echo "MPI PartB (Optimized with -O2)"
  echo "Number of processes: $num_processes"
  srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../../build/src/cpu/mpi_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/smooth-MPI/20K-Smooth-MPI-${num_processes}.jpg
  echo ""
done

# Pthread PartB
mkdir images/smooth-pthread
for num_cores in 1 2 4 8 16 32
do
  echo "Pthread PartB (Optimized with -O2)"
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../../build/src/cpu/pthread_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/smooth-pthread/20K-Smooth-${num_cores}.jpg ${num_cores}
  echo ""
done

# OpenMP PartB
mkdir images/smooth-OpenMP
for num_cores in 1 2 4 8 16 32
do
  echo "OpenMP PartB (Optimized with -O2)"
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../../build/src/cpu/openmp_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/smooth-OpenMP/20K-Smooth-${num_cores}.jpg $num_cores
  echo ""
done

# CUDA PartB
mkdir images/smooth-CUDA
echo "CUDA PartB"
srun -n 1 --gpus 1 ${CURRENT_DIR}/../../build/src/gpu/cuda_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/smooth-CUDA/20K-Smooth.jpg
echo ""

# OpenACC PartB
mkdir images/smooth-openacc
echo "OpenACC PartB"
srun -n 1 --gpus 1 ${CURRENT_DIR}/../../build/src/gpu/openacc_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/smooth-openacc/20K-Smooth.jpg
echo ""