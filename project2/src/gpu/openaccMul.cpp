#include <iostream>
#include <stdexcept>
#include <chrono>
#include "../matrix.hpp"
#include <cstdlib>
#include <openacc.h>

Matrix matrix_multiply_locality(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    Matrix result(M, N);
    
    // Allocate memory for matrices on the GPU
    int* d_mat1 = (int*)malloc(M * K * sizeof(int));
    int* d_mat2 = (int*)malloc(K * N * sizeof(int));
    int* d_result = (int*)malloc(M * N * sizeof(int));

    // Copy matrices from host to GPU
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            d_mat1[i * K + j] = matrix1[i][j];
        }
    }

    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            d_mat2[i * N + j] = matrix2[i][j];
        }
    }

    // Initialize the result matrix on the GPU
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            d_result[i * N + j] = 0;
        }
    }
    
    // Parallelize the matrix multiplication using OpenACC
    #pragma acc parallel loop collapse(2) copyin(d_mat1[0:M*K], d_mat2[0:K*N]) copyout(d_result[0:M*N])
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                d_result[i * N + j] += d_mat1[i * K + k] * d_mat2[k * N + j];
            }
        }
    }

    // Copy the result back from GPU to host
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            result[i][j] = d_result[i * N + j];
        }
    }

    // Free GPU memory
    free(d_mat1);
    free(d_mat2);
    free(d_result);

    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument("Invalid argument, should be: ./executable /path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];
    const std::string matrix2_path = argv[2];
    const std::string result_path = argv[3];
    const int debug_flag = atoi(argv[4]);

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_locality(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;
    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds" << std::endl;

    if (debug_flag == 1){
        std::cout << "Debug Mode" << std::endl;
        // DEBUG THE ANSWER CORRECTNESS
        std::string ans_mat_path;
        if(result.getRows() == 4) ans_mat_path = "results/answers/m12.txt";
        if(result.getRows() == 128) ans_mat_path = "results/answers/m34.txt";
        if(result.getRows() == 1024) ans_mat_path = "results/answers/m56.txt";
        if(result.getRows() == 2048) ans_mat_path = "results/answers/m78.txt";
        if(result.getRows() == 127) ans_mat_path = "results/answers/m910.txt";
        if(result.getRows() == 1818) ans_mat_path = "results/answers/m1112.txt";
        
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

    return 0;
}
