## Requirement

At the outset, you will receive a poorly implemented dense matrix multiplication function, and your task is to **optimize it systematically**, considering factors such as 

- **memory locality, 
- **SIMD (Single Instruction, Multiple Data), 
- **thread-level parallelism, 
- **process-level parallelism** 

In your report, you should **document the performance improvements achieved after applying each technique**. 

Ultimately, you are expected to **submit a program that incorporates all of the aforementioned optimization techniques**, and we will evaluate whether the performance of your implementation meets our expectations.

## [Task1: Memory Locality](https://github.com/tonyyxliu/CSC4005-2023Fall/tree/main/project2#task1-memory-locality)

This implementation suffers from poor performance due to its suboptimal memory locality. In Task 1, you are required to:

1. In your report, analyze why our provided implementation of dense matrix multiplication exhibits such poor performance.
2. Complete the`Matrix matrix_multiply_locality(const Matrix& matrix1, const Matrix& matrix2)`function in `src/locality.cpp`. Your goal is to optimize our implementation by enhancing memory locality.
   1. Note: You **cannot** apply any parallel techniques at this stage.
   2. Hint: Here are some methods you may try to increase memory locality and avoid cache misses:
      1. Change the order of the triple loop
      2. Apply tiled matrix multiplication
3. In your report, demonstrate the performance improvement achieved after implementing your changes.











## [Task2: Data-Level Parallelism](https://github.com/tonyyxliu/CSC4005-2023Fall/tree/main/project2#task2-data-level-parallelism)

After completing Task 1, you should already have a relatively efficient sequential implementation for dense matrix multiplication. However, this level of efficiency is not sufficient. In fact, Single Instruction, Multiple Data (SIMD) techniques have been widely employed in many high-performance matrix multiplication libraries. Therefore, in Task 2, you are tasked with:

1. Completing the `Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2)`function in `src/simd.cpp`. Your goal is to further enhance your implementation of dense matrix multiplication by applying SIMD techniques.
   - **Note: you should build upon the work from Task 1**.
2. In your report, showcasing the performance improvement achieved after implementing these changes













## [Task3: Thread-Level Parallelism](https://github.com/tonyyxliu/CSC4005-2023Fall/tree/main/project2#task3-thread-level-parallelism)

Now is the time to introduce thread-level parallelism to your implementation to further enhance the efficiency of dense matrix multiplication. We recommend utilizing **OpenMP** for its user-friendly approach. Therefore, in Task 3, you have the following objectives:

1. Complete the`Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2)`function in `src/openmp.cpp`. Your goal is to expand the application of thread-level parallelism to your implementation using OpenMP.
   - **Note: You should build upon the work from Task 2.**
2. In your experiments, vary the thread num from 1, 2, 4, 8, 16 to 32, and observe the performance improvements.
3. In your report, showcase the performance improvements achieved after implementing these changes.



















## [Task4: Process-Level Parallelism](https://github.com/tonyyxliu/CSC4005-2023Fall/tree/main/project2#task4-process-level-parallelism)

Finally, you are tasked with introducing process-level parallelism to your dense matrix multiplication implementation to further enhance efficiency, utilizing **MPI**. Therefore, in Task 4, you should:

1. Complete the `Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2)`function in `mpi.cpp`

   to extend the application of process-level parallelism to your implementation using MPI. Additionally, you need to edit the`main`function to incorporate other MPI-specific logic, such as process communication.

   - **Note: You should build upon the work from Task 3.**

2. In your experiments, keep the total thread num fixed at 32 (where `process num` * `thread num per process` = 32), but adjust the process num and thread num per process accordingly (1 x 32, 2 x 16, 4 x 8, 8 x 4, 16 x 2, 32 x 1). Observe the performance changes.

3. In your report, demonstrate the performance improvements (if any) achieved after implementing these changes.

