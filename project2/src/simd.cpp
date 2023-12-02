//
// Created by Zhen Tong on 2023/10/21.
// Email: 120090694@link.cuhk.edu.cn
//
// SIMD + Reordering Matrix Multiplication
//

#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cstdlib>

template<class T> inline void Log(const __m256i & value)
{
    const size_t n = sizeof(__m256i) / sizeof(T);
    T buffer[n];
    _mm256_storeu_si256((__m256i*)buffer, value);
    for (int i = 0; i < n; i++)
        std::cout << buffer[i] << " ";
}

Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    std::cout << "M = " << M << ", N = "<<N << ", K = "<<K << std::endl;

    Matrix result(M, N);
    auto ** memM1 = (float**)malloc((M+16) * sizeof(float*));
    auto ** memM2 = (float**)malloc((K+16) * sizeof(float*));

    for(size_t i = 0; i < M+16; i++){
        // std::cout << i << std::endl;
        memM1[i] = (float*) malloc((K+16)*sizeof(float));
        if(i < M){
            for(size_t j = 0; j < K; j++){
                // std::cout << j << std::endl;
                memM1[i][j] = static_cast< float >(matrix1[i][j]);            
            }
        }
    }
    for(size_t i = 0; i < K+16; i++){
        memM2[i] = (float*)malloc((N+16)*sizeof(float));
        if(i<K){
            for(size_t j = 0; j < N; j++){
                memM2[i][j] = static_cast< float >(matrix2[i][j]);
            }
        }
    }
    auto ** memresult = (float**)malloc((M+16)*sizeof(float*));
    for(size_t i = 0; i < M+16; i++){
        memresult[i] = (float*)malloc((N+16)*sizeof(float));
        for(size_t j = 0; j < N; j++){
            memresult[i][j] = 0.0f;
        }
    }
    
    for(size_t i = 0; i < M; i+=8){
        // load the row pointer of M1 from 1 to 8 into an array
        float * mat1_row_ptrs[] = {
            memM1[i+0],
            memM1[i+1],
            memM1[i+2],
            memM1[i+3],
            memM1[i+4],
            memM1[i+5],
            memM1[i+6],
            memM1[i+7],
        };

        for(size_t k = 0; k < K; k+=8){
            // load the row pointer of M2 from 1 to 8
            auto mat2_ptr_k0 = memM2[k+0];
            auto mat2_ptr_k1 = memM2[k+1];
            auto mat2_ptr_k2 = memM2[k+2];
            auto mat2_ptr_k3 = memM2[k+3];
            auto mat2_ptr_k4 = memM2[k+4];
            auto mat2_ptr_k5 = memM2[k+5];
            auto mat2_ptr_k6 = memM2[k+6];
            auto mat2_ptr_k7 = memM2[k+7];

            for(int x = 0; x < 8; x++){

                for(size_t j = 0; j < N; j += 8){
                    __m256 row = _mm256_setzero_ps();
                    __m256 kv0, kv1 ,kv2, kv3, kv4, kv5, kv6, kv7;
                    kv0 = _mm256_loadu_ps(&mat2_ptr_k0[j]);
                    kv1 = _mm256_loadu_ps(&mat2_ptr_k1[j]);
                    kv2 = _mm256_loadu_ps(&mat2_ptr_k2[j]);
                    kv3 = _mm256_loadu_ps(&mat2_ptr_k3[j]);
                    kv4 = _mm256_loadu_ps(&mat2_ptr_k4[j]);
                    kv5 = _mm256_loadu_ps(&mat2_ptr_k5[j]);
                    kv6 = _mm256_loadu_ps(&mat2_ptr_k6[j]);
                    kv7 = _mm256_loadu_ps(&mat2_ptr_k7[j]);   


                    row += mat1_row_ptrs[x][k+0] * kv0;
                    row += mat1_row_ptrs[x][k+1] * kv1;
                    row += mat1_row_ptrs[x][k+2] * kv2;
                    row += mat1_row_ptrs[x][k+3] * kv3;
                    row += mat1_row_ptrs[x][k+4] * kv4;
                    row += mat1_row_ptrs[x][k+5] * kv5;
                    row += mat1_row_ptrs[x][k+6] * kv6;
                    row += mat1_row_ptrs[x][k+7] * kv7;

                    // load the 1x8 result into sb
                    float sb[8];

                    // Load into a buffer and increase by the tile matrix value
                    _mm256_store_ps(sb, row);

                    for(size_t e = 0; e < 8; e++){
                        memresult[i+x][j+e] += sb[e];
                    }            

                }
            }
        }
    }
    for(int i = 0; i < M; i++){
        auto mem_result_ptr_i = memresult[i];
        for(int j = 0; j < N; j++){
            result[i][j] = mem_result_ptr_i[j];
        }
    }

    for(size_t i = 0; i < M+16; i++){
        free(memM1[i]);
    }
    // transpose M2
    for(size_t i = 0; i < K+16; i++){
        free(memM2[i]);
    }
    for(size_t i = 0; i < M+16; i++){
        free(memresult[i]);
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