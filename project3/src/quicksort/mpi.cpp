//
// Created by Zhen TONG.
// Email: 120090694@link.cuhk.edu.cn
//
// Parallel Quick Sort with MPI
//


#include <iostream>
#include <vector>
#include <mpi.h>
#include <queue>
#include <utility>
#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

void quickSort(std::vector<int> &vec, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        quickSort(vec, low, pivotIndex - 1);
        quickSort(vec, pivotIndex + 1, high);
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
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

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    // divide task
    int num_per_task = size / numtasks;
    int left_num = size % numtasks;
    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_num = 0;
    for (int i = 0; i < numtasks; ++i) {
        if (divided_left_num < left_num) {
            cuts[i+1] = cuts[i] + num_per_task + 1;
            divided_left_num++;
        } else cuts[i+1] = cuts[i] + num_per_task;
    }

    if (taskid == MASTER) {
        // malloc pointer array for answer storing
        int ** rows = (int**)malloc(numtasks*sizeof(int*));
        for(int i = 0; i < numtasks; ++i){
            rows[i] = (int*)malloc((cuts[i+1]-cuts[i])*sizeof(int));
        }

        quickSort(vec, cuts[taskid], cuts[taskid+1]-1);
        auto row = rows[MASTER];
        int row_size = cuts[MASTER+1]-cuts[MASTER];
        for(int i = 0; i < row_size; ++i){
            row[i] = vec[i];
        }

        for (int tid = MASTER + 1; tid < numtasks; ++tid) {
            int length = cuts[tid+1]-cuts[tid];
            MPI_Recv(rows[tid], length, MPI_INT, tid, TAG_GATHER, MPI_COMM_WORLD, &status);
        }


        // TODO Merge answers
        typedef std::pair<int, int> pi; 
        std::priority_queue<pi , std::vector<pi>, std::greater<pi> >compare_q;  
        
        // initialize the queue
        int indcies[numtasks];
        for(int i = 0; i < numtasks; i++){
            auto p0 = std::make_pair(rows[i][0], i);
            compare_q.push(p0);
            indcies[i] = 1;
        }
        
        int index = 0;
        while(!compare_q.empty()){
            auto smallest = compare_q.top();
            // add the smallest element into the vector
            vec[index] = smallest.first;
            index++;
            compare_q.pop();
            int row_num = smallest.second;
            // Check if there are element to add?
            if(indcies[row_num] <cuts[row_num+1]-cuts[row_num]){
                compare_q.push(std::make_pair(rows[row_num][indcies[row_num]], row_num));
                indcies[row_num]+=1;
            }
        }



        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }else{
        quickSort(vec, cuts[taskid], cuts[taskid+1]-1);
        int slave_size = cuts[taskid+1]-cuts[taskid];
        MPI_Send(&vec[cuts[taskid]], slave_size, MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}