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

#define MASTER 0
#define TAG_GATHER 0

void matrix_multiply_mpi(float ** memM1, float ** memM2, \
                                float* buffer, int M, int N, int K,\
                                int start_row, int end_row, int taskid) {
    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to OpenMP, SIMD, Memory Locality and Cache Missing,
    // Further Applying MPI
    // Note:
    // You can change the argument of the function 
    // for your convenience of task division

    std::cout << "FUCK ME " << start_row << std::endl;
    int row_length = end_row-start_row;
    int align = 32;
    auto ** mem_result = (float**)malloc((row_length+16)*sizeof(float*));
    for(size_t i = 0; i < row_length+16; i++){
        mem_result[i] = (float*)_mm_malloc((N/8+1)*8*sizeof(float), align);
        for(size_t j = 0; j < N; j++){
            mem_result[i][j] = 0.0f;
        }
    }

    // #pragma omp parallel for
    for(size_t i = start_row; i < end_row; i++){
        // load the row pointer of M1 from 1 to 8 into an array
        if(taskid == 1){
            std::cout << "row = " << i << std::endl;
        }
        auto mat1_row_ptr = memM1[i];
        __m256 row_vec_i[N/8+1];
        for(int x = 0; x < N/8+1; x++){
            row_vec_i[x] = _mm256_setzero_ps();
        }
        for(size_t k = 0; k < K; k++){
            // auto mat1_ik = mat1_row_ptr[k];
            auto mat1_ik = mat1_row_ptr[k];
            auto mat2_row_ptr = memM2[k];
            for(size_t j = 0; j < N; j+=8){
                __m256 mat2_kj = _mm256_load_ps(&mat2_row_ptr[j]);
                row_vec_i[j/8] += mat1_ik * mat2_kj;
            }
        }
        // load out the row vector into the result
        int cnt_m = 0;

        float * mem_result_row_ptr = mem_result[i];
        for(int y = 0; y < N/8+1; y++){
            _mm256_store_ps(&mem_result_row_ptr[8*y], row_vec_i[y]);
        }
    }

    #pragma omp parallel for
    for(int i = 0; i < row_length; i++){
        auto mem_result_ptr_i = buffer[i];
        for(int j = 0; j < N; j++){
            buffer[i*(N+16)+j] = mem_result[i][j];
        }
    }
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

    // divide task 
    int M = matrix1.getRows(); 
    int N = matrix2.getCols();
    int K = matrix2.getRows();
    std::vector<int> cuts(numtasks+1);
    int row_per_task = M/numtasks;
    int left_row_num = M%numtasks;
    int divide_row_left_num = 0;
    std::cout << "cuts = ";
    for (int i = 0; i < numtasks; ++i) {
        if (divide_row_left_num < left_row_num) {
            cuts[i+1] = cuts[i] + row_per_task + 1;
            divide_row_left_num++;
        } else cuts[i+1] = cuts[i] + row_per_task;
        std::cout << cuts[i] << ", ";
    }
    std::cout << std::endl;

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

    Matrix result(M, N);
    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {
        // initialize the rsult array in heap.
        int align = 32;
        float * buffer = new float[(M+16)*(N+16)];
        matrix_multiply_mpi(memM1, memM2, buffer, M, N, K, cuts[taskid], cuts[taskid+1], taskid);
        for (int i = MASTER + 1; i < numtasks; ++i) {
            auto received_elapsed_time = std::chrono::milliseconds(0);
            int length = (cuts[i+1] - cuts[i]) * (N+16);
            auto start_pos = buffer + cuts[i]*(N+16);
            std::cout << "Receive Length = " << length <<std::endl;
            MPI_Recv(start_pos, length, MPI_FLOAT, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        for(int i = 0; i < (M+16); ++i){
            for(int j = 0; j < (N+16); ++j){
            }
        }
        
        for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
                result[i][j] = buffer[i*(N+16)+j];
                // std::cout << result[i][j] << " ";
            }
            // std::cout << std::endl;
        }
        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
        for(size_t i = 0; i < M+16; i++){
            free(memM1[i]);
        }
        for(size_t i = 0; i < K+16; i++){
            free(memM2[i]);
        }
        free(memM1);
        free(memM2);
        delete[]buffer;

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
        int align = 32;
        int task_row_length = cuts[taskid+1]-cuts[taskid];
        float * sub_buffer = new float[(task_row_length+16)*(N+16)];
        matrix_multiply_mpi(memM1, memM2, sub_buffer, M, N, K, cuts[taskid], cuts[taskid+1], taskid);
        MPI_Send(sub_buffer, task_row_length*(N+16),  MPI_FLOAT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        delete[] sub_buffer;
    }

    MPI_Finalize();

    return 0;
}