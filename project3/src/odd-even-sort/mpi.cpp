//
// Created by Zhen Tong
// Email: 120090694@link.cuhk.edu.cn
//
// Parallel Odd-Even Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0
#define Gather_TAG 1
#define FromLeft_TAG 2
#define FromRight_TAG 3
#define From_Slave_TAG 4
#define From_Master_TAG 5
typedef std::vector<int> vi;


void singleSort(std::vector<int>& vec) {
    bool sorted = false;

    while (!sorted) {
        sorted = true;

        // Perform the odd phase
        for (int i = 1; i < vec.size() - 1; i += 2) {
            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                sorted = false;
            }
        }

        // Perform the even phase
        for (int i = 0; i < vec.size() - 1; i += 2) {
            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                sorted = false;
            }
        }
    }
}


void oddEvenSort(vi& vec, int numtasks, int taskid, vi& cuts, MPI_Status* status) {
    int sorted;
    int begin_even = cuts[taskid]%2==0 ? cuts[taskid] : cuts[taskid]+1;
    int begin_odd = cuts[taskid]%2==1 ? cuts[taskid]: cuts[taskid]+1;


    // std::cout << "<" << cuts[taskid] << ", " << cuts[taskid+1]-1 <<">=";
    // for (int i = cuts[taskid]; i < cuts[taskid+1]; i++){
    //     std::cout << vec[i] << ", ";
    // }
    // std::cout << std::endl;
    int round = 0;
    while (true) {
        // std::cout << "round = " << round << std::endl;
        // std::cout << "<" << cuts[taskid] << ", " << cuts[taskid+1]-1 <<">=";
        // for (int i = cuts[taskid]; i < cuts[taskid+1]; i++){
        //     std::cout << vec[i] << ", ";
        // }
        // std::cout << std::endl;

        round ++;
        sorted = 1;
        // Perform the odd phase
        for (int i = begin_odd; i < cuts[taskid+1] - 1; i += 2) {
            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                sorted = 0;
            }
        }
        // std::cout << "Odd 1, <" << cuts[taskid] << ", " << cuts[taskid+1]-1 <<">, [sort = " << sorted << "]:";
        // for (int i = cuts[taskid]; i < cuts[taskid+1]; i++){
        //     std::cout << vec[i] << ", ";
        // }

        // communicate swap odd phase
        int last_index = cuts[taskid+1]-1;

        if(cuts[taskid]%2==0 && taskid>0){
            MPI_Send(&vec[cuts[taskid]], 1, MPI_INT, taskid-1, FromRight_TAG, MPI_COMM_WORLD);
            int fromLeft;
            MPI_Recv(&fromLeft, 1, MPI_INT, taskid-1, FromLeft_TAG, MPI_COMM_WORLD, status);
            if(fromLeft != vec[cuts[taskid]]){
                vec[cuts[taskid]] = fromLeft;
                sorted = 0;
            }
        }

        if(last_index%2==1 && taskid < numtasks-1){
            int fromRight;
            MPI_Recv(&fromRight, 1, MPI_INT, taskid+1, FromRight_TAG, MPI_COMM_WORLD, status);
            if(fromRight < vec[last_index]){
                int lastVal = vec[last_index];
                vec[last_index] = fromRight;
                MPI_Send(&lastVal, 1, MPI_INT, taskid+1, FromLeft_TAG, MPI_COMM_WORLD);
                sorted = 0;

            }else{
                MPI_Send(&fromRight, 1, MPI_INT, taskid+1, FromLeft_TAG, MPI_COMM_WORLD);
            }
        }

        // std::cout << "Odd 2, <" << cuts[taskid] << ", " << cuts[taskid+1]-1 <<">, [sort = " << sorted << "]:";
        // for (int i = cuts[taskid]; i < cuts[taskid+1]; i++){
        //     std::cout << vec[i] << ", ";
        // }


        // Perform the even phase
        for (int i = begin_even; i < cuts[taskid+1] - 1; i += 2) {
            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                sorted = 0;
            }
        }
        // std::cout << "Even 1, <" << cuts[taskid] << ", " << cuts[taskid+1]-1 <<">, [sort = " << sorted << "]:";
        // for (int i = cuts[taskid]; i < cuts[taskid+1]; i++){
        //     std::cout << vec[i] << ", ";
        // }


        if(cuts[taskid]%2 == 1 && taskid>0){
            MPI_Send(&vec[cuts[taskid]], 1, MPI_INT, taskid-1, FromRight_TAG, MPI_COMM_WORLD);
            int fromLeft;
            MPI_Recv(&fromLeft, 1, MPI_INT, taskid-1, FromLeft_TAG, MPI_COMM_WORLD, status);
            if(fromLeft != vec[cuts[taskid]]){
                vec[cuts[taskid]] = fromLeft;
                sorted = 0;
            }
        }

        if(last_index % 2==0 && taskid < numtasks-1){
            int fromRight;
            MPI_Recv(&fromRight, 1, MPI_INT, taskid+1, FromRight_TAG, MPI_COMM_WORLD, status);
            if(fromRight < vec[last_index]){
                int lastVal = vec[last_index];
                vec[last_index] = fromRight;
                MPI_Send(&lastVal, 1, MPI_INT, taskid+1, FromLeft_TAG, MPI_COMM_WORLD);
                sorted = 0;
            }else{
                MPI_Send(&fromRight, 1, MPI_INT, taskid+1, FromLeft_TAG, MPI_COMM_WORLD);
            }
        }
        // std::cout << "Even 2, <" << cuts[taskid] << ", " << cuts[taskid+1]-1 <<">, [sort = " << sorted << "]:";
        // for (int i = cuts[taskid]; i < cuts[taskid+1]; i++){
        //     std::cout << vec[i] << ", ";
        // }


        // send end signal
        if(taskid == MASTER){
            int recv_sorted;
            int global_sorted = sorted;
            // std::cout << "tid = Master" << ", sort = " << recv_sorted << std::endl;
            for(int tid = MASTER+1; tid < numtasks; tid++){
                MPI_Recv(&recv_sorted, 1, MPI_INT, tid, From_Slave_TAG, MPI_COMM_WORLD, status);
                global_sorted *= recv_sorted;
                // std::cout << "tid = " << tid << ", sort = " << recv_sorted << std::endl;
            }
            for(int tid = MASTER+1; tid < numtasks; tid++){
                MPI_Send(&global_sorted, 1, MPI_INT, tid, From_Master_TAG, MPI_COMM_WORLD);
            }
            if(global_sorted == 1){
                break;
            }
        }else{
            int global_sorted;
            MPI_Send(&sorted, 1, MPI_INT, MASTER, From_Slave_TAG, MPI_COMM_WORLD);
            MPI_Recv(&global_sorted, 1, MPI_INT, MASTER, From_Master_TAG, MPI_COMM_WORLD, status);
            if(global_sorted == 1){
                break;
            }
        }
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

    int num_per_task = size/numtasks;
    int num_left = size%numtasks;
    int divided_number_left = 0;
    std::vector<int> cuts(numtasks+1, 0);
    for(int i = 0; i < numtasks; ++i){
        if(divided_number_left<num_left){
            cuts[i+1] = cuts[i] + num_per_task + 1;
            divided_number_left ++;
        }else{
            cuts[i+1] = cuts[i] + num_per_task;
        }
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();

    if(numtasks == 1){
        singleSort(vec);
    }
    else{
        oddEvenSort(vec, numtasks, taskid, cuts, &status);
    }

    if (taskid == MASTER) {
        for(int tid = MASTER+1; tid < numtasks; tid++){
            int * wrt_in_ptr = &vec[cuts[tid]];
            MPI_Recv(wrt_in_ptr, cuts[tid+1]-cuts[tid], MPI_INT, tid, Gather_TAG, MPI_COMM_WORLD, &status);
        }

        int index = 0;
        // std::cout << "Final Check Answer" << std::endl;
        // for(int i = 0; i < numtasks; ++i){
        //     std::cout << "<" << cuts[i] << ", " << cuts[i+1]-1 << ">="; 
        //     for (int j = 0; j < cuts[i+1]-cuts[i]; j++){
        //         std::cout << vec[index++] << ", ";
        //     }
        //     std::cout << std::endl;
        // }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Odd-Even Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }else{
        MPI_Send(&vec[cuts[taskid]], cuts[taskid+1]-cuts[taskid], MPI_INT, MASTER, Gather_TAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}