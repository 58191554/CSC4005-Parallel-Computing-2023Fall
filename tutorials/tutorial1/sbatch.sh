#!/bin/bash
#SBATCH -o ./result.out
#SBATCH -p Debug
#SBATCH -J Tutorial1-Sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

# Compile the programs before execution
# srun hostname
mpic++ mpi_vector_addition.cpp -o mpi_vector_addition
g++ -o openmp_vector_addition openmp_vector_addition.cpp -fopenmp
g++ pthread_vector_addition.cpp -o pthread_vector_addition -lpthread
nvcc -o  cuda_vector_addition cuda_vector_addition.cu 
# Four different processes for MPI (Multi-Process Program)
srun -n 4 --mpi=pmi2 ./mpi_vector_addition
# One task, four threads (Multi-Thread Program)
srun -n 1 --cpus-per-task 4 ./openmp_vector_addition
# One task, four threads (Multi-Thread Program)
srun -n 1 --cpus-per-task 4 ./pthread_vector_addition
# One task, with one GPU card
srun -n 1 --gpus 1 ./cuda_vector_addition
