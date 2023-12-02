//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>

#define MASTER 0
#define TAG_GATHER 0

Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, \
                            int threads_num, std::vector<int> cuts, int taskid) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }
    int K = matrix1.getCols(), N = matrix2.getCols();
    int row_length = cuts[taskid+1]-cuts[taskid];

    Matrix result(row_length, N);

    #pragma omp parallel for schedule(static)
    for(size_t i =  cuts[taskid]; i <  cuts[taskid+1]; i++){
        // load the row pointer of M1 from 1 to 8 into an array
        auto mat1_row_ptr = matrix1[i];
        __m256i  row_vec_i[N/8+1];
        for(int x = 0; x < N/8+1; x++){
            row_vec_i[x] = _mm256_setzero_si256();
        }
        for(size_t k = 0; k < K; k++){
            // auto mat1_ik = mat1_row_ptr[k];
            auto mat1_ik = mat1_row_ptr[k];
            auto mat2_row_ptr = matrix2[k];
            __m256i mat1_i_vec = _mm256_set1_epi32(mat1_ik);
            for(size_t j = 0; j < N; j+=8){
                __m256i  mat2_kj = _mm256_loadu_si256((__m256i*)&mat2_row_ptr[j]);
                row_vec_i[j/8] = _mm256_add_epi32(_mm256_mullo_epi32(mat1_i_vec, mat2_kj), row_vec_i[j/8]);
            }
        }
        // load out the row vector into the result

        int * mem_result_row_ptr = result[i-cuts[taskid]];
        for(int y = 0; y < N/8+1; y++){
            _mm256_storeu_si256((__m256i*)&mem_result_row_ptr[y*8], row_vec_i[y]);
        }
    }

    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 6) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    const int debug = atoi(argv[5]);

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    int M = matrix1.getRows(); int N = matrix2.getCols(); int K = matrix1.getCols();

    // divide task
    int row_num_per_task = M / numtasks;
    int left_row_num = M % numtasks;
    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_row_num = 0;
    for (int i = 0; i < numtasks; ++i) {
        if (divided_left_row_num < left_row_num) {
            cuts[i+1] = cuts[i] + row_num_per_task + 1;
            divided_left_row_num++;
        } else cuts[i+1] = cuts[i] + row_num_per_task;
    }


    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {

        Matrix result(M, N);

        Matrix sub_result = matrix_multiply_mpi(matrix1, matrix2, thread_num, cuts, taskid);
        int * buffer = new int[M*N];
        for(int i = cuts[MASTER]; i < cuts[MASTER+1]; i++){
            for(int j = 0; j < N; j++){
                result[i][j] = sub_result[i][j];
                // std::cout << mem_result[i][j] << ", ";
            }
            // std::cout << std::endl;
        }
        for (int tid = MASTER + 1; tid < numtasks; ++tid) {
            auto received_elapsed_time = std::chrono::milliseconds(0);
            int * write_in_pos = buffer+cuts[tid]*N;
            int length = (cuts[tid+1] - cuts[tid]) * N;
            MPI_Recv(write_in_pos, length, MPI_INT, tid, TAG_GATHER, MPI_COMM_WORLD, &status);
            // std::cout << "Child Process " << tid << " Recieve Time: " << received_elapsed_time.count() << " milliseconds\n";
        }

        // std::cout << "Write to the matirx result = "<< std::endl;
        for(int i = cuts[MASTER+1]; i < M; i++){
            for(int j = 0; j < N; j++){
                result[i][j] = buffer[i*N+j];
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;

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

    } else {
        int row_length = cuts[taskid+1]-cuts[taskid];
        int * buffer = new int[row_length*N];
        Matrix sub_result = matrix_multiply_mpi(matrix1, matrix2, thread_num, cuts, taskid);
        for(int i = 0; i < cuts[taskid+1]-cuts[taskid];i++){
            for(int j = 0; j < N; j ++){
                buffer[i*N+j] = sub_result[i][j];
            }
        }
        MPI_Send(buffer, row_length*N, MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        // for (int i = 0; i < 100; i++) {
        //     memcpy(matrix[i], sub_result[i], N * sizeof(int));
        // }
    }

    MPI_Finalize();
    return 0;
}