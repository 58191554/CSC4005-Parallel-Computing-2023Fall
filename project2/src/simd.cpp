//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// SIMD + Reordering Matrix Multiplication
//

#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cstdlib>

Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);
    // malloc with 2d pointer array
    auto ** memM1 = (int**)malloc(M*sizeof(int*));
    for(int i = 0; i < M; i++){
        memM1[i] = (int*)malloc(K*sizeof(int));
        for(int j = 0; j < K; j++){
            memM1[i][j] = matrix1[i][j];
        }
    }
    auto ** memM2 = (int **)malloc(K*sizeof(int*));
    for(int i = 0; i < K; i++){
        memM2[i] = (int*)malloc(N*sizeof(int));
        for(int j =0;j < N; j++){
            memM2[i][j] = matrix2[i][j];
        }
    }
    auto ** memR = (int**)malloc(M*sizeof(int*));
    for(int i = 0 ; i < M; i++){
        memR[i] = (int*)malloc(N*sizeof(int));
        for(int j = 0; j < N; j++){
            memR[i][j] = 0;
        }
    }
    // Robust
    size_t tile_size_M = (32 > M) ? 32 : M;
    size_t tile_size_K = (32 > K) ? 32 : K;
    size_t tile_size_N = (32 > N) ? 32 : N;

    int num_tile_M = M/tile_size_M + 1;
    int num_tile_K = K/tile_size_K + 1;
    int num_tile_N = N/tile_size_N + 1;

    // tiled matrix multiplication
    for(int ti = 0; ti < num_tile_M; ++ti){
        for(int tj = 0; tj < num_tile_N; ++tj){
            for(int tk = 0; tk < num_tile_K; ++tk){
                size_t offset_M = ti*tile_size_M;
                size_t offset_K = tk*tile_size_K;
                size_t offset_N = tj*tile_size_N;
                size_t bound_M = ((ti+1)*tile_size_M < M) ? (ti+1)*tile_size_M : M;
                size_t bound_N = ((tj+1)*tile_size_N < N) ? (tj+1)*tile_size_N : N;
                size_t bound_K = ((tk+1)*tile_size_K < K) ? (tk+1)*tile_size_K : K;
                for(size_t i = offset_M; i < bound_M; i++){
                    for(size_t j = offset_N; j < bound_N; j++){
                        for(size_t k = offset_K; k < bound_K; k++){
                            memR[i][j] += memM1[i][k]*memM2[k][j];
                        }
                    }
                }
            }
        }
    }
    

    // load the answer
    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < N; j++){
            result[i][j] = memR[i][j];
        }
    }



    // free the heap
    for(int i = 0; i < M; i++){
        free(memM1[i]);
    }
    for(int i = 0; i < K; i++){
        free(memM2[i]);
    }
    for(int i = 0; i < M; i++){
        free(memR[i]);
    }

    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    const int debug = atoi(argv[4]);

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_simd(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    if (debug == 1){
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