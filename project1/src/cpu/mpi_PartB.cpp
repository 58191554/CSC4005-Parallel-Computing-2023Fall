//
// Created by Zhen TONG on 2023/9/29.
// Email: 120090694@link.cuhk.edu.cn
//
// MPI implementation of average filter
//

#include <iostream>
#include <vector>
#include <chrono>

#include <mpi.h>    // MPI Header

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0
int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
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


    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);


    // Divide the tasks
    int total_pixel_num = (input_jpeg.width-2) * (input_jpeg.height-2);
    int pixel_num_per_task = total_pixel_num / numtasks;
    int left_pixel_num = total_pixel_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_pixel_num = 0;

    for (int i = 0; i < numtasks; ++i) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            divided_left_pixel_num++;
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }

    auto rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }


    auto start_time = std::chrono::high_resolution_clock::now();
    if(taskid == MASTER){
        // master process
        auto filterImage = new unsigned char[input_jpeg.width*input_jpeg.height*input_jpeg.num_channels];

        // master do the kernel work for the first cut
        int start_row = cuts[taskid]/(input_jpeg.width-2);
        int end_row = (cuts[taskid+1]-1)/(input_jpeg.width-2)+1;
        for(int i = start_row; i < end_row; ++i){
            for(int j = 2; j < input_jpeg.width-1; ++j){
                int cuts_index = (i-1)*(input_jpeg.width-2)+(j-1);
                if(cuts_index <cuts[taskid+1] && cuts[taskid]<=cuts_index){
                    int indices[9] = {
                        (i-1)*input_jpeg.width + j - 1,
                        (i-1)*input_jpeg.width + j,
                        (i-1)*input_jpeg.width + j + 1,
                        (i)*input_jpeg.width + j - 1,
                        (i)*input_jpeg.width + j,
                        (i)*input_jpeg.width + j + 1,
                        (i+1)*input_jpeg.width + j - 1,
                        (i+1)*input_jpeg.width + j,
                        (i+1)*input_jpeg.width + j + 1
                    };
                    unsigned char r = 0;
                    unsigned char g = 0;
                    unsigned char b = 0;
                    for(int k = 0; k < 9; ++k){
                        r += rChannel[indices[k]]/9;
                        g += gChannel[indices[k]]/9;
                        b += bChannel[indices[k]]/9;
                    }
                    filterImage[(cuts_index-cuts[taskid])*input_jpeg.num_channels  ] = r;
                    filterImage[(cuts_index-cuts[taskid])*input_jpeg.num_channels+1] = g;
                    filterImage[(cuts_index-cuts[taskid])*input_jpeg.num_channels+2] = b;       
                }
            }
        }
        
        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; ++i) {
            auto received_elapsed_time = std::chrono::milliseconds(0);
            unsigned char* start_pos = filterImage + cuts[i]*input_jpeg.num_channels;
            int length = (cuts[i+1] - cuts[i])*input_jpeg.num_channels;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            std::cout << "Child Process " << i << " Recieve Time: " << received_elapsed_time.count() << " milliseconds\n";
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // Save output JPEG image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filterImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }


        // heap resource delete
        delete[] filterImage;
        delete[] rChannel;
        delete[] gChannel;
        delete[] bChannel;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }else{
        // slave process
        // auto son_start_time = std::chrono::high_resolution_clock::now();
        int cut_len = cuts[taskid+1]-cuts[taskid];
        auto filterImageSeg = new unsigned char[cut_len*input_jpeg.num_channels];
        int start_row = cuts[taskid]/(input_jpeg.width-2);
        int end_row = (cuts[taskid+1]-1)/(input_jpeg.width-2)+1;
        for(int i = start_row; i < end_row; ++i){
            for(int j = 2; j < input_jpeg.width-1; ++j){
                int cuts_index = (i-1)*(input_jpeg.width-2)+(j-1);
                if(cuts_index <cuts[taskid+1] && cuts[taskid]<=cuts_index){
                    int indices[9] = {
                        (i-1)*input_jpeg.width + j - 1,
                        (i-1)*input_jpeg.width + j,
                        (i-1)*input_jpeg.width + j + 1,
                        (i)*input_jpeg.width + j - 1,
                        (i)*input_jpeg.width + j,
                        (i)*input_jpeg.width + j + 1,
                        (i+1)*input_jpeg.width + j - 1,
                        (i+1)*input_jpeg.width + j,
                        (i+1)*input_jpeg.width + j + 1
                    };
                    unsigned char r = 0;
                    unsigned char g = 0;
                    unsigned char b = 0;
                    for(int k = 0; k < 9; ++k){
                        r += rChannel[indices[k]]/9;
                        g += gChannel[indices[k]]/9;
                        b += bChannel[indices[k]]/9;
                    }
                    filterImageSeg[(cuts_index-cuts[taskid])*input_jpeg.num_channels  ] = r;
                    filterImageSeg[(cuts_index-cuts[taskid])*input_jpeg.num_channels+1] = g;
                    filterImageSeg[(cuts_index-cuts[taskid])*input_jpeg.num_channels+2] = b;       
                }
            }
        }


        // Send the filter segmentation to the master process
        MPI_Send(filterImageSeg, cut_len*input_jpeg.num_channels,  MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        // auto son_end_time = std::chrono::high_resolution_clock::now();
        // auto son_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(son_end_time - start_time);
        // std::cout << "Son Complete!" << std::endl;
        // std::cout << "Execution Time: " << son_elapsed_time.count() << " milliseconds\n";
        // Release heap source
        delete[] filterImageSeg;
    }
    return 0;
}