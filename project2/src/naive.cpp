//
// Created by Zhen Tong
// Email: 120090694@link.cuhk.edu.cn
//
// Naive Matrix Multiplication
//

#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

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

Matrix matrix_multiply(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);
    // get the max gcd as the tile size
    size_t tile_size = gcd(M, gcd(K, N));
    size_t tile_num_M = M/tile_size;
    size_t tile_num_N = N/tile_size;
    size_t tile_num_K = K/tile_size;
    for(size_t ti = 0; ti < tile_num_M; ++ti){
        for(size_t tj = 0; tj < tile_num_N; ++tj){
            for(size_t tk = 0; tk < tile_num_K; ++tk){
                // do the tile matrix multiply for tile_num_K x tile_num_K times
                // for each tile matrix multiplication
                size_t row_offset = ti * tile_size;
                size_t col_offset = tj * tile_size;
                size_t mid_offset = tk * tile_size;
                for(size_t i = 0; i < tile_size; ++i){
                    for(size_t j = 0; j < tile_size; ++j){
                        for(size_t k = 0; k < tile_size; k++){
                            result[row_offset + i][col_offset + j] += 
                            matrix1[row_offset + i][mid_offset + k] * matrix2[mid_offset + k][col_offset + j];
                        }
                    }
                }
            }
        }
    }



    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}