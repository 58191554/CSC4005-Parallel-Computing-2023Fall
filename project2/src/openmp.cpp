//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// OpenMp + SIMD + Reordering Matrix Multiplication
//scan

#include <immintrin.h>
#include <omp.h> 
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cstdlib>

Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2, int num_threads) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

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
    auto ** mem_result = (float**)malloc((M+16)*sizeof(float*));
    for(size_t i = 0; i < M+16; i++){
        mem_result[i] = (float*)malloc((N/8+1)*8*sizeof(float));
        for(size_t j = 0; j < N; j++){
            mem_result[i][j] = 0.0f;
        }
    }

    #pragma omp parallel for
    for(size_t i = 0; i < M; i++){
        // load the row pointer of M1 from 1 to 8 into an array
        auto mat1_row_ptr = memM1[i];
        __m256 row_vec_i[N/8+1];
        for(int x = 0; x < N/8+1; x++){
            row_vec_i[x] = _mm256_setzero_ps();
        }
        for(size_t k = 0; k < K; k++){
            // auto mat1_ik = mat1_row_ptr[k];
            auto mat1_ik = mat1_row_ptr[k];
            auto mat2_row_ptr = memM2[k];
            __m256 mat1_i_vec = _mm256_set1_ps(mat1_ik);
            for(size_t j = 0; j < N; j+=8){
                __m256 mat2_kj = _mm256_load_ps(&mat2_row_ptr[j]);
                row_vec_i[j/8] = _mm256_add_ps(_mm256_mul_ps(mat1_i_vec, mat2_kj), row_vec_i[j/8]);
            }
        }
        // load out the row vector into the result
        int cnt_m = 0;
        mem_result[i] = (float*)malloc((N/8+1)*8*sizeof(float));
        for(size_t x = 0; x < N; x++){
            mem_result[i][x] = 0.0f;
        }

        float * mem_result_row_ptr = mem_result[i];
        for(int y = 0; y < N/8+1; y++){

            float sb[8];
            _mm256_store_ps(sb, row_vec_i[y]);
            for(int z = 0; z < 8; z++){
                if(cnt_m < M){
                    mem_result_row_ptr[cnt_m] = sb[cnt_m%8];
                    cnt_m++;
                }
            }
        }
    }

    #pragma parallel for
    for(int i = 0; i < M; i++){
        auto mem_result_ptr_i = mem_result[i];
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
        free(mem_result[i]);
    }
    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 6) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable num_threads"
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    int num_threads = atoi(argv[1]);

    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    const int debug = atoi(argv[5]);

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_openmp(matrix1, matrix2, num_threads);

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