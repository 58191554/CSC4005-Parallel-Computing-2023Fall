//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Odd-Even Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0
#define From_Left_TAG 2
#define From_Right_TAG 3
#define From_Master 4
#define From_Slave 5

void singleOddEvenSort(std::vector<int>& vec) {
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


bool oddEvenSort(int * vec, int sid, int oddeven, std::vector<int> &cuts) {
    // return the subdomain updated and need to do the sort again
    bool update = false;
    int index;
    if(oddeven%2 == 0){
        if(cuts[sid] % 2 == 0){
            index = 0;
        }
        else{
            index = 1;
        }
    }else{
        if(cuts[sid]%2 == 1){
            index = 0;
        }
        else{
            index = 1;
        }
    }
    for(; index+1 < cuts[sid+1]-cuts[sid]; index+=2){
        int a = vec[index];
        int b = vec[index+1];
        if(b<a){
            vec[index] = b;
            vec[index+1] = a;
            update = true;
        }
    }
    return update;
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

    // divide task
    int numslaves = numtasks==1 ? 1 : numtasks-1;
    int num_per_slave = size/numslaves;
    int num_left = size%numslaves;
    int divided_number_left = 0;
    std::vector<int> cuts(numslaves+1, 0);
    for(int i = 0; i < numslaves; ++i){
        if(divided_number_left<num_left){
            cuts[i+1] = cuts[i] + num_per_slave + 1;
            divided_number_left ++;
        }else{
            cuts[i+1] = cuts[i] + num_per_slave;
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();


    if (taskid == MASTER) {

        const int seed = 4005;

        std::vector<int> vec = createRandomVec(size, seed);
        std::vector<int> vec_clone = vec;


        if(numtasks == 1){
            singleOddEvenSort(vec);
        }
        else{
            int load_index = 0;
            int** task_data = (int**)malloc((numslaves)*sizeof(int*));
            for(int i = 0; i < numslaves; i++){
                int cut_length = cuts[i+1]-cuts[i];
                // the message: (signal) + vec_cut
                task_data[i] = (int*)malloc((1+cut_length)*sizeof(int));
                auto message = task_data[i];
                for(int j = 1; j < cut_length + 1; j++){
                    message[j] = vec[load_index++];
                }
            }

            // for(int i = 0; i < numslaves; i++){
            //     auto results = task_data[i];
            //     int cut_size = cuts[i+1]-cuts[i];
            //     std::cout << "<" << cuts[i] << ", " << cuts[i+1]-1 << "> = ";
            //     for(int j = 0; j < cut_size; j++){
            //         std::cout << results[j+1] << ", ";
            //     }
            //     std::cout << std::endl;
            // }


            bool sorted = false;
            int round = 0;
            while (sorted != true)
            {
                sorted = true;
                // Check edge exchange 
                // std::cout << "Odd Or Even? " << round%2 << std::endl;
                for (int sid = 0; sid < numslaves-1; sid++){
                    // |a1...., b1|, |a2, ..., b2|, |a3, ..., b3|, ...
                    int b1_index = cuts[sid+1]-cuts[sid]-1;
                    auto vec1 = &task_data[sid][1];
                    auto vec2 = &task_data[sid+1][1];
                    if((round%2)==((cuts[sid+1]-1)%2)){
                        // edge case
                        int b1 = vec1[b1_index];
                        int a2 = vec2[0];
                        // std::cout << "exchange data: " << b1 <<", " <<a2<<std::endl;
                        if (b1 > a2){
                            vec1[b1_index] = a2;
                            vec2[0] = b1;
                            sorted = sorted && false;
                        }
                    }
                }
                // Construct msg and send
                for(int tid = 1; tid < numtasks; tid++){
                    int sid = tid-1;
                    int cut_length = cuts[sid+1]-cuts[sid];
                    int msg_length = cut_length+1;
                    int * msg = task_data[sid];
                    msg[0] = round % 2;
                    MPI_Send(msg, msg_length, MPI_INT, tid, From_Master, MPI_COMM_WORLD);
                }

                // Collect the sort state, and update the task_data
                for(int tid = 1; tid < numtasks; tid++){
                    int sid = tid-1;
                    auto wrt_in_position = task_data[sid];
                    int cut_length = cuts[sid+1]-cuts[sid];
                    int msg_length = cut_length+1;
                    MPI_Recv(wrt_in_position, msg_length, MPI_INT, tid, From_Slave, MPI_COMM_WORLD, &status);
                    if(wrt_in_position[0] == 1){
                        sorted = sorted && false;
                    }
                }

                // std::cout << "CHECK Answer" << std::endl;
                // for(int i = 0; i < numslaves; i++){
                //     auto results = task_data[i];
                //     int cut_size = cuts[i+1]-cuts[i];
                //     std::cout << "<" << cuts[i] << ", " << cuts[i+1]-1 << "> = ";
                //     for(int j = 0; j < cut_size; j++){
                //         std::cout << results[j+1] << ", ";
                //     }
                //     std::cout << std::endl;
                // }
                round++;
            }
            // Communicate with slave let them stop
            int stop = -1;
            for(int tid = MASTER+1; tid < numtasks; tid++){
                MPI_Send(&stop, 1, MPI_INT, tid, From_Master, MPI_COMM_WORLD);                
            }

            // load the result in the vec
            int index = 0;
            // std::cout << "Final Check" << std::endl;
            for(int i = 0; i < numslaves; ++i){
                auto result = task_data[i];
                int cut_length = cuts[i+1]-cuts[i];
                for(int j = 0; j < cut_length; j++){
                    vec[index] = result[j+1];
                    // std::cout << result[j+1] << ", ";
                    index++;
                }
                // std::cout << std::endl;
            }
        }       

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Odd-Even Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }else{
    // [SLAVE MODE]
        int sid = taskid-1;
        int left_idx = cuts[sid];
        int right_idx = cuts[sid+1]-1;
        int cut_length = cuts[sid+1]-cuts[sid];
        int msg_length = cut_length+1;
        // message = (oddevenSignal) + vec_cut
        int * message = (int*)malloc((cut_length+1)*sizeof(int));
        int round = 0;
        while(true){
            MPI_Recv(message, msg_length, MPI_INT, MASTER, From_Master, MPI_COMM_WORLD, &status);
            if(message[0] == -1){
                break;
            }
            int oddeven = message[0]; 
            int * vec = &message[1];
            bool update = false;

            update = update || oddEvenSort(vec, sid, oddeven, cuts);
            message[0] = update ? 1 : 0;
            MPI_Send(message, msg_length, MPI_INT, MASTER, From_Slave, MPI_COMM_WORLD);
            round ++;
        }
        free(message);
    }

    MPI_Finalize();
    return 0;
}       