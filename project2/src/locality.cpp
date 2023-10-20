//
// Created by Zhen Tong
// Email: 120090694@link.cuhk.edu.cn
//
// Naive Matrix Multiplication
//

#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cstdlib>

Matrix tile_multiply(Matrix& tile_M1, Matrix& tile_M2, int tile_size){
    Matrix tile_result(tile_size, tile_size);
    for (size_t i = 0; i < tile_size; ++i) {
        for (size_t j = 0; j < tile_size; ++j) {
            for (size_t k = 0; k < tile_size; ++k) {
                tile_result[i][j] += tile_M1[i][k] * tile_M2[k][j];
            }
        }
    }
    return tile_result;
}

size_t gcd(size_t a, size_t b) {
    while (b != 0) {
        size_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

Matrix matrix_multiply_locality(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    Matrix result(M, N);
    auto * memM1 = (int*)malloc(M*K*sizeof(int));
    auto * memM2 = (int*)malloc(K*N*sizeof(int));

    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < K; j++){
            memM1[i*M+j] = matrix1[i][j];            
        }
    }
    for(size_t i = 0; i < K; i++){
        for(size_t j = 0; j < N; j++){
            memM2[i*K + j] = matrix2[i][j];
        }
    }
    auto * memresult = (int*)malloc(M*N*sizeof(int));
    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < N; j++){
            memresult[i*M+j] = 0;
        }
    }
    std::cout << "FUCK ME" << std::endl;

    // get the max gcd as the tile size
    // size_t tile_size = gcd(M, gcd(K, N));
    size_t tile_sizeM = 32;
    size_t tile_sizeN = 32;
    size_t tile_sizeK = 32;
    if(tile_sizeM > M){
        tile_sizeM = M;
    }
    if(tile_sizeN > N){
        tile_sizeN = N;
    }
    if(tile_sizeK > K){
        tile_sizeK = K;
    }

    std::cout << "M = " << M << ", N = "<<N << ", K = "<<K << std::endl;
    std::cout << "tile_sizeM = " << tile_sizeM 
    << "tile_sizeN = " << tile_sizeN
    << "tile_sizeK = " << tile_sizeK  << std::endl;
    
    size_t tile_num_M = M/tile_sizeM;
    size_t tile_num_N = N/tile_sizeN;
    size_t tile_num_K = K/tile_sizeK;
    // 1. Change the order of the tripple nested loop
    // 2. Apply Tiled Matrix Multiplication

    // tiled
    for(size_t ti = 0; ti < tile_num_M; ++ti){
        for(size_t tj = 0; tj < tile_num_N; ++tj){
            for(size_t tk = 0; tk < tile_num_K; ++tk){
                // do the tile matrix multiply for tile_num_K x tile_num_K times
                // for each tile matrix multiplication
                size_t row_offset = ti * tile_sizeM;
                size_t col_offset = tj * tile_sizeN;
                size_t mid_offset = tk * tile_sizeK;
                for(size_t i = 0; i < tile_sizeM; ++i){
                    for(size_t j = 0; j < tile_sizeN; ++j){
                        for(size_t k = 0; k < tile_sizeK; k++){
                            // std::cout <<"i = "<<i<<" j = "<<j<<" k = "<<k<<std::endl;
                            // std::cout << "memM1:" << memM1[(row_offset + i)*M + (mid_offset + k)]<< " * memM2:"
                            // << memM2[(mid_offset + k)*K + (col_offset + j)];
                            memresult[(row_offset + i)*M+(col_offset+j)] += 
                            memM1[(row_offset + i)*M + (mid_offset + k)] *
                            memM2[(mid_offset + k)*K + (col_offset + j)];
                            // std::cout << " reuslt:" <<memresult[(row_offset + i)*M+(col_offset+j)] <<std::endl;
                        }
                    }
                }
            }
        }
    }

    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < N; j++){
            result[i][j] = memresult[i*M+j];
        }
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

    const int debug_flag = atoi(argv[4]);

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_locality(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    if (debug_flag == 1){
        std::cout << "Debug Mode" << std::endl;
        // DEBUG THE ANSWER CORRECTNESS
        std::string ans_mat_path;
        if(result.getRows() == 4) ans_mat_path = "results/answers/m12.txt";
        if(result.getRows() == 128) ans_mat_path = "results/answers/m34.txt";
        if(result.getRows() == 1024) ans_mat_path = "results/answers/m56.txt";
        if(result.getRows() == 2048) ans_mat_path = "results/answers/m78.txt";
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