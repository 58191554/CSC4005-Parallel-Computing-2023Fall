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

int oddEvenSort(int * vec, int sid, int oddeven, std::vector<int> &cuts) {
    // return the subdomain updated and need to do the sort again
    int update = 0;
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
    for(; index < cuts[sid+1]-cuts[sid]; index+=2){
        int a = vec[index];
        int b = vec[index+1];
        if(b<a){
            vec[index] = b;
            vec[index+1] = a;
            update = 1;
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
    int numslaves = numtasks-1;
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

        for(int i = 0; i < size; i ++){
            std::cout << vec[i] << ", ";
        }
        std::cout << std::endl;




        int load_index = 0;
        int** task_data = (int**)malloc((numslaves)*sizeof(int*));
        for(int i = 0; i < numslaves; i++){
            int cut_length = cuts[i+1]-cuts[i];
            // the message: (signal) + vec_cut
            task_data[i] = (int*)malloc((1+cut_length)*sizeof(int));
            auto message = task_data[i];
            std::cout << "Index range = <" << cuts[i] << ", " << cuts[i+1]-1 << ">: ";
            for(int j = 1; j < cut_length + 1; j++){
                message[j] = vec[load_index++];
                std::cout << message[j] <<", ";
            }
            std::cout << std::endl;
        }

        bool sorted = false;
        int round = 0;
        while (sorted != true)
        {
            std::cout << "ROUND = " << round << std::endl;
            sorted = true;
            // Construct msg and send
            for(int tid = MASTER+1; tid<numtasks; tid++){
                int sid = tid-1;
                int cut_length = cuts[sid+1]-cuts[sid];
                int msg_length = cut_length + 1;
                int * msg = task_data[sid];
                // add the odd/even signal
                msg[0] = round%2;

                MPI_Send(msg, msg_length, MPI_INT, tid, From_Master, MPI_COMM_WORLD);                
            }

            // Collect the sort state, and update the task_data
            for(int tid = MASTER+1; tid<numtasks; tid++){

                int sid = tid-1;
                auto write_in_msg = task_data[sid];
                int cut_length = cuts[sid+1]-cuts[sid];
                int msg_length = cut_length + 1;
                // receive data, and update data

                MPI_Recv(write_in_msg, msg_length, MPI_INT, tid, From_Slave, MPI_COMM_WORLD, &status);
                // check if next sort is needed.
                if(write_in_msg[0] == 1){
                    sorted = false;
                }
            }


            std::cout << "Check Answer after One Iteration" << std::endl;
            for(int i = 0; i < numslaves; ++i){
                auto result = task_data[i];
                int cut_length = cuts[i+1]-cuts[i];
                std::cout << "Index range = <" << cuts[i] << ", " << cuts[i+1]-1 << ">: ";

                for(int j = 0; j < cut_length; j++){
                    std::cout << result[j+1] <<", ";
                }
                std::cout << std::endl;

            }

            round++;
        }
        // Communicate with slave let them stop
        int stop = -1;
        for(int tid = MASTER+1; tid < numtasks; tid++){
            MPI_Send(&stop, 1, MPI_INT, tid, From_Master, MPI_COMM_WORLD);                
        }
        
        // load the result in the vec
        std::cout << "Check Answer!"<<std::endl;
        int index = 0;
        for(int i = 0; i < numslaves; ++i){
            auto result = task_data[i];
            int cut_length = cuts[i+1]-cuts[i];
            std::cout << "Index range = <" << cuts[i] << ", " << cuts[i+1]-1 << ">: ";

            for(int j = 0; j < cut_length; j++){
                std::cout << result[j+1] <<", ";
                vec[index] = result[j+1];
                index++;
            }
            std::cout << std::endl;

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
        while(true){
            MPI_Recv(message, msg_length, MPI_INT, MASTER, From_Master, MPI_COMM_WORLD, &status);
            if(message[0] == -1){
                break;
            }
            int oddeven = message[0]; 
            int * vec = &message[1];

            // check whether the right-most element need to communicate with slave right
            bool rightComm = false;
            if(oddeven%2 == 0){
                if(right_idx%2 == 0){
                    rightComm = true;
                }
            }else{
                if(right_idx%2 == 1){
                    rightComm = true;
                }
            }
            // Communicate with the right slave
            if(rightComm && taskid<numtasks-1){
                int from_right_val;
                std::cout << "taskid = " << taskid << ", state = " << oddeven << ", Communicate with the RIGHT slave"<< std::endl;
                MPI_Recv(&from_right_val, 1, MPI_INT, taskid+1, From_Right_TAG, MPI_COMM_WORLD, &status);
                // compare value
                if(vec[cut_length-1] < from_right_val){
                    // Do nothing and send back
                    MPI_Send(&from_right_val, 1, MPI_INT, taskid+1, From_Left_TAG, MPI_COMM_WORLD);
                }
                else{
                    MPI_Send(&vec[cut_length-1], 1, MPI_INT, taskid+1, From_Left_TAG, MPI_COMM_WORLD);
                    vec[cut_length-1] = from_right_val;
                }
            }

            // check whether the left-most element need to communicate with slave left
            bool leftComm = false;
            if(oddeven == 0){
                // if state is doing even state,
                if(left_idx%2 == 1){
                    leftComm = true;
                }
            }else{
                // if state is doing odd state
                if(left_idx%2 == 0){
                    leftComm = true;
                }
            }
            // communicate with left slave
            if(taskid!=1 && leftComm){
                int from_left_val;
                std::cout << "taskid = " << taskid << ", state = " << oddeven << ", Communicate with the LEFT slave"<< std::endl;
                MPI_Send(&vec[0], 1, MPI_INT, taskid-1, From_Right_TAG, MPI_COMM_WORLD);
                // Received the compared 
                MPI_Recv(&from_left_val, 1, MPI_INT, taskid-1, From_Left_TAG, MPI_COMM_WORLD, &status);
                vec[0] = from_left_val;
            }
            message[0] = oddEvenSort(vec, sid, oddeven, cuts);
            MPI_Send(message, msg_length, MPI_INT, MASTER, From_Slave, MPI_COMM_WORLD);

        }
        free(message);
    }

    MPI_Finalize();
    return 0;
}       