//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0
typedef std::vector<int> vi; 

void insertionSort(vi& bucket) {
    for (int i = 1; i < bucket.size(); ++i) {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key) {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

int bucketSort(vi& vec, vi &cuts, int num_per_bucket, int taskid, int * row) {

    // Pre-allocate space to avoid re-allocation
    int num_buckets = cuts[taskid+1]-cuts[taskid];
    std::vector<std::vector<int>> buckets(num_buckets);
    // Pre-allocate space to avoid re-allocation
    for (std::vector<int>& bucket : buckets) {
        bucket.reserve(num_per_bucket);
    }

    int min_bucket_id = cuts[taskid];
    int max_bucket_id = cuts[taskid+1]-1;
    // Place each element in the appropriate bucket
    for(int num: vec){
        int bucket_id = num/num_per_bucket;
        // check if the num is in the range of the slave buckets
        if(bucket_id >= min_bucket_id && bucket_id <= max_bucket_id){
            buckets[bucket_id].push_back(num);
        }
    }
    // Sort each bucket using insertion sort
    for (std::vector<int>& bucket : buckets) {
        insertionSort(bucket);
    }
    int index = 0;
    for (const std::vector<int>& bucket : buckets) {
        for (int num : bucket) {
            row[index++] = num;
        }
    }
    return index;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size bucket_num\n"
            );
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

    const int size = atoi(argv[1]);

    const int bucket_num = atoi(argv[2]);

    const int seed = 4005;

    vi vec = createRandomVec(size, seed);
    vi vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());
    int range = max_val - min_val + 1;
    // divide bucket
    int num_per_bucket = range/bucket_num+1;
    int num_redundent = num_per_bucket*bucket_num-range;

    // divide task
    int buckets_per_task = bucket_num/numtasks;
    int buckets_left = bucket_num%numtasks;
    int divided_bucket_left = 0;
    vi cuts(numtasks+1, 0);
    for(int i = 0; i < numtasks; ++i){
        if(divided_bucket_left<buckets_left){
            cuts[i+1] = cuts[i] + buckets_per_task + 1;
            divided_bucket_left ++;
        }
        else{
            cuts[i+1] = cuts[i] + buckets_per_task;
        }
    }

    if (taskid == MASTER) {
        int ** rows = (int**)malloc(numtasks*sizeof(int*));
        for(int i = 0; i < numtasks; ++i){
            int row_length = (cuts[i+1]-cuts[i])*num_per_bucket;
            rows[i] = (int*) malloc(row_length*sizeof(int));
        }
        int sort_sizes[numtasks];
        bucketSort(vec, cuts, num_per_bucket, taskid, rows[MASTER]);
        for (int tid = MASTER + 1; tid < numtasks; ++tid) {
            int * write_in_pos = rows[tid];
            int length = (cuts[tid+1] - cuts[tid]) * num_per_bucket;
            MPI_Recv(write_in_pos, length, MPI_INT, tid, TAG_GATHER, MPI_COMM_WORLD, &status);
            MPI_Recv(&sort_sizes[tid], 1, MPI_INT, tid, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        std::cout << "FUCK ME" << std::endl;

        // load the data from the int array to the vector
        int index = 0;
        for(int i = 0; i < numtasks; i++){
            auto row = rows[i];
            for(int j = 0; j < sort_sizes[i]; ++j){
                vec[i*num_per_bucket+j] = row[j];
                index++;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
        for(int i = 0; i < numtasks; ++i){
            free(rows[i]);
        }        
        free(rows);
    }else{
        int range = (cuts[taskid+1]-cuts[taskid])*num_per_bucket;
        int * row = (int*)malloc(range*sizeof(int));
        int row_size = bucketSort(vec, cuts, num_per_bucket, taskid, row);
        MPI_Send(row, range, MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        MPI_Send(&row_size, 1, MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}