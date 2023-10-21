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

    Matrix result(M, N);
    auto ** memM1 = (int**)malloc(M * sizeof(int*));
    // transpose M2
    auto ** memM2 = (int**)malloc(N * sizeof(int*));

    for(size_t i = 0; i < M; i++){
        // std::cout << i << std::endl;
        memM1[i] = (int*) malloc(K*sizeof(int));
        for(size_t j = 0; j < K; j++){
            // std::cout << j << std::endl;
            memM1[i][j] = matrix1[i][j];            
        }
    }
    // transpose M2
    for(size_t i = 0; i < N; i++){
        memM2[i] = (int*)malloc(K*sizeof(int));
        for(size_t j = 0; j < K; j++){
            memM2[i][j] = matrix2[j][i];
        }
    }
    auto ** memresult = (int**)malloc(M*sizeof(int*));
    for(size_t i = 0; i < M; i++){
        memresult[i] = (int*)malloc(N*sizeof(int));
        for(size_t j = 0; j < N; j++){
            memresult[i][j] = 0;
        }
    }

    // get the max gcd as the tile size
    // size_t tile_size = gcd(M, gcd(K, N));
    size_t tile_sizeM = 8;
    size_t tile_sizeN = 8;
    size_t tile_sizeK = 8;
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
    
    size_t numtile_M = M/tile_sizeM+1;
    size_t numtile_N = N/tile_sizeN+1;
    size_t numtile_K = K/tile_sizeK+1;
    // store arrays of tile sizes
    int* tile_sizes_M = (int*)malloc(numtile_M*sizeof(int));
    int* tile_sizes_N = (int*)malloc(numtile_N*sizeof(int));
    int* tile_sizes_K = (int*)malloc(numtile_K*sizeof(int));

    for(int i = 0; i < numtile_M; i++){
        if(i < numtile_M-1) tile_sizes_M[i] = tile_sizeM;
        else tile_sizes_M[i] = M-i*tile_sizeM;
    }

    for(int i = 0; i < numtile_N; i++){
        if(i<numtile_N-1)tile_sizes_N[i] = tile_sizeN;
        else tile_sizes_N[i] = N-i*tile_sizeN;
    }
    for(int i = 0; i < numtile_K; i++){
        if(i < numtile_K-1) tile_sizes_K[i] = tile_sizeK;
        else tile_sizes_K[i] = K-i*tile_sizeK;
    }


    // 1. Change the order of the tripple nested loop
    // 2. Apply Tiled Matrix Multiplication

    // tiled
    for(size_t ti = 0; ti < numtile_M; ++ti){
        for(size_t tj = 0; tj < numtile_N; ++tj){
            for(size_t tk = 0; tk < numtile_K; ++tk){
                // do the tile matrix multiply for tile_num_K x tile_num_K times
                size_t row_offset = ti * tile_sizeM;
                size_t col_offset = tj * tile_sizeN;
                size_t mid_offset = tk * tile_sizeK;

                size_t lenM = tile_sizes_M[ti];
                size_t lenN = tile_sizes_N[tj];
                size_t lenK = tile_sizes_K[tk];
                // load a row of 8 int into the __m128i
                for(size_t i = 0; i < lenM; i++){
                    for(size_t j = 0; j < lenN; j++){
                        int m1_arr[16];
                        int m2_arr[16];
                        for(size_t k =0; k < lenK; k++){
                            m1_arr[k] = memM1[row_offset+i][mid_offset+k];
                            m2_arr[k] = memM2[col_offset+j][mid_offset+k];
                            std::cout << m1_arr[k] <<" ";
                            std::cout << m2_arr[k] <<" ";

                        }
                        std::cout << std::endl;
                        std::cout << "row, col = "<< row_offset+i << " " << col_offset+j << std::endl;
                        __m256i m1si = _mm256_load_si256((__m256i*)(&m1_arr[0]));
                        __m256i m2si = _mm256_load_si256((__m256i*)(&m2_arr[0])); // transpose
                        __m256i rlow  = _mm256_mul_epi32(m1si, m2si);
                        // No high??
                        int rs_arr[8];
                        _mm256_store_si256((__m256i*)(&rs_arr), rlow);
                        std::cout << "FUCK ME" << std::endl;
                        for(size_t k = 0; k < 8; k++ ){
                            std::cout << rs_arr[k] <<" ";
                        }
                        std::cout << std::endl;

                    }
                }
            }
        }
    }

    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < N; j++){
            result[i][j] = memresult[i][j];
        }
    }
    for(size_t i = 0; i < M; i++){
        free(memM1[i]);
    }
    // transpose M2
    for(size_t i = 0; i < N; i++){
        free(memM2[i]);
    }
    for(size_t i = 0; i < M; i++){
        free(memresult[i]);
    }
    delete[] tile_sizes_M;
    delete[] tile_sizes_N;
    delete[] tile_sizes_K;
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