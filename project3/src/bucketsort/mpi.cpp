//
// Created by Zhen Tong 
// Email: 120090694@link.cuhk.edu.cn
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

void bucketSort(vi& vec, vi &cuts, int min_num, int size, int num_per_bucket, int taskid, int * result) {
    // put number into slave buckets, sort, and report the number in the buckets.
    int b_min = min_num + cuts[taskid] * num_per_bucket;
    int b_max = min_num + cuts[taskid+1]*num_per_bucket;
    int num_buckets = cuts[taskid+1]-cuts[taskid];
    std::vector<std::vector<int>> buckets(num_buckets);
    // std::cout << "max = " << b_max << std::endl;
    // std::cout << "min = " << b_min << std::endl;
    for(int num: vec){
        if(num<b_max && num >= b_min){
            int bi = (num-b_min)/num_per_bucket;
            // std::cout << num << ", " ;
            buckets[bi].push_back(num);
        }
    }
    // std::cout <<std::endl;
    for (std::vector<int>& bucket : buckets) {
        insertionSort(bucket);
    }
    // load the sorted data to the row
    int index = 1;
    for (const std::vector<int>& bucket : buckets) {
        for (int num : bucket) {
            result[index++] = num;
        }
    }
    // for(int i = 0; i < index; ++i){
    //     std::cout << row[i] << ", ";
    // }
    // std::cout << "sorted number = " << index<< std::endl;
    result[0] = index-1;
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

    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());
    int range = max_val - min_val + 1;
    // divide bucket, num_per_bucket larger the precise 
    int num_per_bucket = range/bucket_num + 1;
    // divide task: from bucket 0 ~ bucket_num, uniform bucket
    int bucket_per_task = bucket_num/numtasks;
    int bucket_left = bucket_num%numtasks;
    int divide_bucket_left = 0;
    vi cuts(numtasks+1, 0);
    for(int i = 0; i < numtasks; i++){
        if(divide_bucket_left<bucket_left){
            cuts[i+1] = cuts[i] + bucket_per_task + 1;
            divide_bucket_left++;
        }
        else{
            cuts[i+1] = cuts[i] + bucket_per_task;
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();


    if (taskid == MASTER) {
        int ** task_data = (int**)malloc(numtasks*sizeof(int*));
        for(int i = 0; i < numtasks; i++){
            // the bucketed sort can be extremely concentrate
            // Set the first element in array as size header
            task_data[i] = (int*) malloc((1+size)*sizeof(int));
        }
        bucketSort(vec, cuts, min_val, size, num_per_bucket, MASTER, task_data[MASTER]);

        for (int tid = MASTER + 1; tid < numtasks; ++tid) {
            int * write_in_pos = task_data[tid];
            MPI_Recv(write_in_pos, size+1, MPI_INT, tid, TAG_GATHER, MPI_COMM_WORLD, &status);
        }
        // load the data from the int array to the vector
        int index = 0;
        for(int tid = 0; tid < numtasks; ++tid){
            auto result = task_data[tid];
            int sort_size = result[0];
            // std::cout << "result size" << sorted_size[tid] << std::endl;
            for(int j = 1; j < sort_size+1; j++){
                // std::cout << "j = " << j << std::endl;
                vec[index++] = result[j];
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;

        checkSortResult(vec_clone, vec);
        for(int i = 0; i < numtasks; i++){
            free(task_data[i]);
        }
        free(task_data);
    }else{
        // set the first int in result as the size header
        int * result = (int*)malloc((1+size)*sizeof(int));
        bucketSort(vec, cuts, min_val, size, num_per_bucket, taskid, result);
        MPI_Send(result, result[0]+1, MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        free(result);
    }

    MPI_Finalize();
    return 0;
}