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

    std::cout << "M = " << M << ", N = "<<N << ", K = "<<K << std::endl;

    Matrix result(M, N);
    auto ** memM1 = (float**)malloc((M) * sizeof(float*));
    auto ** memM2 = (float**)malloc((K) * sizeof(float*));

    for(size_t i = 0; i < M; i++){
        // std::cout << i << std::endl;
        memM1[i] = (float*) malloc((K)*sizeof(float));
        if(i < M){
            for(size_t j = 0; j < K; j++){
                // std::cout << j << std::endl;
                if(j < K){
                    memM1[i][j] = static_cast< float >(matrix1[i][j]);   
                }         
            }
        }
    }
    for(size_t i = 0; i < K; i++){
        memM2[i] = (float*)malloc((N)*sizeof(float));
        if(i<K){
            for(size_t j = 0; j < N; j++){
                if(j < N){
                    memM2[i][j] = static_cast< float >(matrix2[i][j]);
                }
            }
        }
    }
    auto ** memresult = (float**)malloc((M)*sizeof(float*));
    for(size_t i = 0; i < M; i++){
        memresult[i] = (float*)malloc((N)*sizeof(float));
        for(size_t j = 0; j < N; j++){
            memresult[i][j] = 0.0f;
        }
    }
    
    for(size_t i = 0; i < M; ++i){
        auto mat1_ptr_i = memM1[i];
        auto mem_result_ptr_i = memresult[i];
        for(size_t k = 0; k < K; ++k){
            auto mat2_ptr_k = memM2[k];
            auto mat1_ik = mat1_ptr_i[k];
            for(size_t j = 0; j < N; ++j){
                mem_result_ptr_i[j] += mat1_ik*mat2_ptr_k[j];
            }
        }

    }

    for(int i = 0; i < M; i++){
        auto mem_result_ptr_i = memresult[i];
        for(int j = 0; j < N; j++){
            result[i][j] = mem_result_ptr_i[j];
        }
    }

    for(size_t i = 0; i < M; i++){
        free(memM1[i]);
    }
    // transpose M2
    for(size_t i = 0; i < K; i++){
        free(memM2[i]);
    }
    for(size_t i = 0; i < M; i++){
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