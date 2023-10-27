//
// Created by Zhen Tong on 2023/10/07.
// Email: 120090694@link.cuhk.edu.cn
//
// Matrix Multiplication with CUDA, for bonus
//

#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>
#include <chrono>
#include "../matrix.hpp"
#include <cstdlib>

__global__ void matrix_multiply_cuda(int* d_mat1, int* d_mat2, int* d_result,\
                                        int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        int value = 0;
        for (int k = 0; k < K; k++) {
            value += d_mat1[row*K+k] * d_mat2[k*N+col];
        }
        d_result[row*N+col] = value;
    }
}

int main(int argc, char** argv) {
    int deviceID = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    std::cout << "Max block size for device " << deviceID << ": " << deviceProp.maxThreadsPerBlock << std::endl;

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    const int debug = atoi(argv[4]);

    // load the matrix
    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);
    int M = matrix1.getRows(); int N = matrix2.getCols(); int K = matrix1.getCols();
    Matrix result(M, N);

    int* d_mat1; // Device-side array of pointers
    int* d_mat2; // Device-side array of pointers
    int* d_result; // Device-side array of pointers
    
    cudaMalloc((void**)&d_mat1, M * K * sizeof(int));
    for (int i = 0; i < M; i++) {
        cudaMemcpy(&d_mat1[i*K], matrix1[i], K * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&d_mat2, K * N * sizeof(int));
    for (int i = 0; i < K; i++) {
        cudaMemcpy(&d_mat2[i*N], matrix2[i], N * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&d_result, M * N * sizeof(int));
   
    
    // Record time
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); // GPU start time
    int thread_num_x = 16/* Choose an appropriate number of threads per block */;
    int thread_num_y = 16;

    // Determine the grid dimension.
    int blocksPerGridY = (N + thread_num_y - 1) / thread_num_y;
    int blocksPerGridX = (M + thread_num_x - 1) / thread_num_x;

    dim3 blockDim(thread_num_x, thread_num_y);
    dim3 gridDim(blocksPerGridX, blocksPerGridY);    
    matrix_multiply_cuda<<<gridDim, blockDim>>>(d_mat1, d_mat2, d_result, M, N, K);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);


    // copy data from host to device
    for (int i = 0; i < M; i++) {
        cudaMemcpy(result[i], &d_result[i*N], N * sizeof(int), cudaMemcpyDeviceToHost);
    }


    // Free Cuda memory
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);


    result.saveToFile(result_path);

    if (debug == 1){
        std::cout << "Debug Mode" << std::endl;
        // DEBUG THE ANSWER CORRECTNESS
        std::string ans_mat_path;
        if(result.getRows() == 4) ans_mat_path = "results/answers/m12.txt";
        else if(result.getRows() == 128) ans_mat_path = "results/answers/m34.txt";
        else if(result.getRows() == 1024) ans_mat_path = "results/answers/m56.txt";
        else if(result.getRows() == 2048) ans_mat_path = "results/answers/m78.txt";
        else if(result.getRows() == 127) ans_mat_path = "results/answers/m910.txt";
        else if(result.getRows() == 1818) ans_mat_path = "results/answers/m1112.txt";
        else if(result.getRows() == 3) ans_mat_path = "results/answers/mab.txt";
        else if(result.getRows() == 1) ans_mat_path = "results/answers/mcd.txt";
        else return 0;
        
        std::cout << "ans_mat_path = " << ans_mat_path << std::endl;
        Matrix matrix_ans = Matrix::loadFromFile(ans_mat_path);

        bool correct = true;
        for(size_t i = 0; i < result.getRows(); ++i){
            if (correct){
                for(size_t j = 0; j < result.getCols(); ++j){ 
                    if(result[i][j] != matrix_ans[i][j]){
                        std::cout << "wrong index i = "<<i<<" j = "<<j<<std::endl;
                        std::cout << result[i][j] << " != " << matrix_ans[i][j]<<std::endl;
                        correct = false;
                        break;
                    }
                }
            }
        }
        if(correct){
            std::cout << "Answer Correct!" << std::endl;
        }
        // DEBUG THE ANSWER CORRECTNESS ENDS
    }
    std::cout << "Matrix Multiply Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
