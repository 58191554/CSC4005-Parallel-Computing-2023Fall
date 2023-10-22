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

    // set tile size
    size_t tile_size = 32;
    std::cout << "M = " << M << ", N = "<<N << ", K = "<<K << std::endl;
    std::cout << "tile_sizeM = " << tile_size 
    << " tile_sizeN = " << tile_size
    << " tile_sizeK = " << tile_size  << std::endl;


    Matrix result(M, N);
    auto ** memM1 = (float**)malloc((M+tile_size) * sizeof(float*));
    auto ** memM2 = (float**)malloc((K+tile_size) * sizeof(float*));

    for(size_t i = 0; i < M+tile_size; i++){
        // std::cout << i << std::endl;
        memM1[i] = (float*) malloc((K+tile_size)*sizeof(float));
        if(i < M){
            for(size_t j = 0; j < K+8; j++){
                // std::cout << j << std::endl;
                if(j < K){
                    memM1[i][j] = static_cast< float >(matrix1[i][j]);   
                }         
            }
        }
    }
    for(size_t i = 0; i < K+tile_size; i++){
        memM2[i] = (float*)malloc((N+8)*sizeof(float));
        if(i<K){
            for(size_t j = 0; j < N+tile_size; j++){
                if(j < N){
                    memM2[i][j] = static_cast< float >(matrix2[i][j]);
                }
            }
        }
    }
    auto ** memresult = (float**)malloc((M+tile_size)*sizeof(float*));
    for(size_t i = 0; i < M+8; i++){
        memresult[i] = (float*)malloc((N+tile_size)*sizeof(float));
        for(size_t j = 0; j < N; j++){
            memresult[i][j] = 0.0f;
        }
    }
    
    size_t numtile_M = M/tile_size+1;
    size_t numtile_N = N/tile_size+1;
    size_t numtile_K = K/tile_size+1;
    // store arrays of tile sizes
    int* tile_sizes_M = (int*)malloc(numtile_M*sizeof(int));
    int* tile_sizes_N = (int*)malloc(numtile_N*sizeof(int));
    int* tile_sizes_K = (int*)malloc(numtile_K*sizeof(int));

    // get length for each valid tile
    for(int i = 0; i < numtile_M; i++){
        if(i < numtile_M-1) tile_sizes_M[i] = tile_size;
        else tile_sizes_M[i] = M-i*tile_size;
    }
    for(int i = 0; i < numtile_N; i++){
        if(i<numtile_N-1)tile_sizes_N[i] = tile_size;
        else tile_sizes_N[i] = N-i*tile_size;
    }
    for(int i = 0; i < numtile_K; i++){
        if(i < numtile_K-1) tile_sizes_K[i] = tile_size;
        else tile_sizes_K[i] = K-i*tile_size;
    }

    // 1. Change the order of the tripple nested loop
    // 2. Apply Tiled Matrix Multiplication
    // tiled
    for(size_t ti = 0; ti < numtile_M; ++ti){

        for(size_t tj = 0; tj < numtile_N; ++tj){

            for(size_t tk = 0; tk < numtile_K; ++tk){
                // do the tile matrix multiply for tile_num_K x tile_num_K times
                size_t row_offset = ti * tile_size;
                size_t col_offset = tj * tile_size;
                size_t mid_offset = tk * tile_size;

                size_t lenM = tile_sizes_M[ti];
                size_t lenN = tile_sizes_N[tj];
                size_t lenK = tile_sizes_K[tk];
                // std::cout << "tile info: ti, tj, tk = "<<ti << " " << tj << " "<< tk << std::endl;
                // In each tile multiply compute divide into 2x2 vectorized compute
                for(size_t x = 0; x < lenM; x+=8){
                    for(size_t y = 0; y < lenN; y+=8){
                        for(size_t i = 0; i < 8; i++){
                            // take a 1x8 row and do vectorization and add vector to get a 1x8 result
                            // take the next 1x8 row horizontaly and do the vector compute again
                            for(size_t z = 0; z < lenK;z += 8){
                                // std::cout << "x, y, z, i = " <<x<< ", "<<y<<", "<<z <<", "<<i << std::endl;
                                // declear the registors
                                __m256 row = _mm256_setzero_ps();
                                __m256 k0, k1 ,k2, k3, k4, k5, k6, k7;
                                k0 = _mm256_loadu_ps(&memM2[mid_offset+z+0][col_offset+y]);
                                k1 = _mm256_loadu_ps(&memM2[mid_offset+z+1][col_offset+y]);
                                k2 = _mm256_loadu_ps(&memM2[mid_offset+z+2][col_offset+y]);
                                k3 = _mm256_loadu_ps(&memM2[mid_offset+z+3][col_offset+y]);
                                k4 = _mm256_loadu_ps(&memM2[mid_offset+z+4][col_offset+y]);
                                k5 = _mm256_loadu_ps(&memM2[mid_offset+z+5][col_offset+y]);
                                k6 = _mm256_loadu_ps(&memM2[mid_offset+z+6][col_offset+y]);
                                k7 = _mm256_loadu_ps(&memM2[mid_offset+z+7][col_offset+y]);
                                row += memM1[row_offset+x+i][mid_offset+z+0] * k0;
                                row += memM1[row_offset+x+i][mid_offset+z+1] * k1;
                                row += memM1[row_offset+x+i][mid_offset+z+2] * k2;
                                row += memM1[row_offset+x+i][mid_offset+z+3] * k3;
                                row += memM1[row_offset+x+i][mid_offset+z+4] * k4;
                                row += memM1[row_offset+x+i][mid_offset+z+5] * k5;
                                row += memM1[row_offset+x+i][mid_offset+z+6] * k6;
                                row += memM1[row_offset+x+i][mid_offset+z+7] * k7;
                                // load the 1x8 result into sb
                                float sb[8];
                                // Load into a buffer and increase by the tile matrix value
                                _mm256_store_ps(sb, row);

                                for(size_t e = 0; e < 8; e++){
                                    memresult[row_offset+x+i][col_offset+y+e] += sb[e];
                                }
                            }
                        }       
                    }
                }
            }
            for(size_t i = ti*tile_size; i < ti*tile_size+tile_sizes_M[ti]; i++){
                for(size_t j = tj*tile_size; j < tj*tile_size+tile_sizes_N[tj]; j++){
                    result[i][j] = static_cast<int>(memresult[i][j]);
                }
            }
        }
    }


    for(size_t i = 0; i < M+8; i++){
        free(memM1[i]);
    }
    // transpose M2
    for(size_t i = 0; i < K+8; i++){
        free(memM2[i]);
    }
    for(size_t i = 0; i < M+8; i++){
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