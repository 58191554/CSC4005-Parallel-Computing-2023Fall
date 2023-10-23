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
    auto ** memresult = (float**)malloc((M+16)*sizeof(float*));
    for(size_t i = 0; i < M+16; i++){
        memresult[i] = (float*)malloc((N+16)*sizeof(float));
        for(size_t j = 0; j < N; j++){
            memresult[i][j] = 0.0f;
        }
    }
    
    // divide task
    int M_pad = (M%8 == 0) ? M : (M/8) * 8 + 8;
    int N_pad = (N%8 == 0) ? N : (N/8) * 8 + 8;
    std::cout << "M_pad, N_pad = " << M_pad << ", " << N_pad << std::endl;
    int total_task_num = M_pad*N_pad / 64;
    int block_rows_num = M_pad/8;
    int block_cols_num = N_pad/8;

    int block_per_task = total_task_num/num_threads;
    int left_block_num = total_task_num%num_threads;

    std::vector <int> cuts(num_threads+1, 0);
    int divided_left_block_num = 0;
    int actual_num_thread = 0;

    for (int i = 0; i < num_threads; ++i) {
        if (divided_left_block_num < left_block_num) {
            cuts[i+1] = cuts[i] + block_per_task + 1;
            divided_left_block_num++;
        } else cuts[i+1] = cuts[i] + block_per_task;
        if(cuts[i]!=cuts[i+1]){
            actual_num_thread++;
        }
    }
    num_threads = actual_num_thread;
    std::cout << "actural threads number = " << num_threads << std::endl;
    omp_set_num_threads(num_threads);
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel  shared(memM1, memM2, memresult, cuts, M_pad, N_pad,\
    num_threads, result, block_rows_num, block_cols_num)
    #pragma omp for
    for(int task_id = 0; task_id < num_threads; task_id++){
        // for each task, parse the linear task array to 2d tile index
        // std::cout << "taskid = " << task_id  <<std::endl;
        // parse the block_id to 2d array
        int block_num = cuts[task_id+1]-cuts[task_id];

        for(int block_id = cuts[task_id]; block_id < cuts[task_id+1]; block_id++){
            size_t block_row = (block_id/block_cols_num)*8;
            size_t block_col = (block_id-block_row*block_cols_num/8)*8;
            // std::cout << "block_id, block_row, block_col = " << block_id << ", "
            //  << block_row << ", " << block_col << std::endl;

            for(int x = 0; x < 8; x++){
                // laod the row pointer of the mat1
                auto mat1_row_ptr = memM1[block_row + x];
                auto mem_result_row_ptr = memresult[block_row+x];

                // load the row pointer of M2 from k+0 to k+7 (k incrs by 8)
                for(size_t k = 0; k < K; k += 8){
                    auto mat2_ptr_k0 = memM2[k+0];
                    auto mat2_ptr_k1 = memM2[k+1];
                    auto mat2_ptr_k2 = memM2[k+2];
                    auto mat2_ptr_k3 = memM2[k+3];
                    auto mat2_ptr_k4 = memM2[k+4];
                    auto mat2_ptr_k5 = memM2[k+5];
                    auto mat2_ptr_k6 = memM2[k+6];
                    auto mat2_ptr_k7 = memM2[k+7];

                    __m256 row = _mm256_setzero_ps();
                    __m256 kv0, kv1 ,kv2, kv3, kv4, kv5, kv6, kv7;
                    kv0 = _mm256_loadu_ps(&mat2_ptr_k0[block_col]);
                    kv1 = _mm256_loadu_ps(&mat2_ptr_k1[block_col]);
                    kv2 = _mm256_loadu_ps(&mat2_ptr_k2[block_col]);
                    kv3 = _mm256_loadu_ps(&mat2_ptr_k3[block_col]);
                    kv4 = _mm256_loadu_ps(&mat2_ptr_k4[block_col]);
                    kv5 = _mm256_loadu_ps(&mat2_ptr_k5[block_col]);
                    kv6 = _mm256_loadu_ps(&mat2_ptr_k6[block_col]);
                    kv7 = _mm256_loadu_ps(&mat2_ptr_k7[block_col]);   


                    row += mat1_row_ptr[k+0] * kv0;
                    row += mat1_row_ptr[k+1] * kv1;
                    row += mat1_row_ptr[k+2] * kv2;
                    row += mat1_row_ptr[k+3] * kv3;
                    row += mat1_row_ptr[k+4] * kv4;
                    row += mat1_row_ptr[k+5] * kv5;
                    row += mat1_row_ptr[k+6] * kv6;
                    row += mat1_row_ptr[k+7] * kv7;

                    // load the 1x8 result into sb
                    float sb[8];

                    // Load into a buffer and increase by the tile matrix value
                    _mm256_store_ps(sb, row);
                    for(size_t e = 0; e < 8; e++){
                        mem_result_row_ptr[block_col+e] += sb[e];
                    }            
                    
                }
            }
        }
    }
        
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "MultiThread Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    std::cout <<"FUCK ME" << std::endl;
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