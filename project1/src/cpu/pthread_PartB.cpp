//
// Created by Zhen TONG on 2023/9/30.
// Email: 120090694@link.cuhk.edu.cn
//
// Pthread implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <chrono>
#include <pthread.h>
#include "utils.hpp"

// Structure to pass data to each thread
struct ThreadData {
    unsigned char* rChannels;
    unsigned char* gChannels;
    unsigned char* bChannels;
    unsigned char* output_buffer;
    int start;
    int end;
    int width;
    int height;
    int num_channels;
};

// Convolution Kernel
void* conv_func(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    
    // TODO
    int start_row = data->start/(data->width-2);
    int end_row = (data->end)/(data->width-2)+1;
    for(int i = start_row; i < end_row; ++i){
        for(int j = 2; j < data->width-1; ++j){
            int cuts_index = (i-1)*(data->width-2)+(j-1);
            if(cuts_index <=data->end && data->start<=cuts_index){
                int indices[9] = {
                    (i-1)*data->width + j - 1,
                    (i-1)*data->width + j,
                    (i-1)*data->width + j + 1,
                    (i)*data->width + j - 1,
                    (i)*data->width + j,
                    (i)*data->width + j + 1,
                    (i+1)*data->width + j - 1,
                    (i+1)*data->width + j,
                    (i+1)*data->width + j + 1
                };
                unsigned char r = 0;
                unsigned char g = 0;
                unsigned char b = 0;
                for(int k = 0; k < 9; ++k){
                    r += data->rChannels[indices[k]]/9;
                    g += data->gChannels[indices[k]]/9;
                    b += data->bChannels[indices[k]]/9;
                }
                data->output_buffer[cuts_index*data->num_channels  ] = r;
                data->output_buffer[cuts_index*data->num_channels+1] = g;
                data->output_buffer[cuts_index*data->num_channels+2] = b;       
            }
        }
    }
        

    return nullptr;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads\n";
        return -1;
    }

    int num_threads = std::stoi(argv[3]); // User-specified thread count

    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);

    auto filterImage = new unsigned char[(input_jpeg.width-2) * (input_jpeg.height-2) * input_jpeg.num_channels];
    
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    for(int i = 0; i < num_threads; i++){
        thread_data[i].width = input_jpeg.width;
        thread_data[i].height = input_jpeg.height;
        thread_data[i].num_channels = input_jpeg.num_channels;
    }

    // load data to 3 channels
    auto rChannels = new unsigned char[input_jpeg.width*input_jpeg.height];
    auto gChannels = new unsigned char[input_jpeg.width*input_jpeg.height];
    auto bChannels = new unsigned char[input_jpeg.width*input_jpeg.height];
    
    for(int i = 0; i < input_jpeg.width*input_jpeg.height; ++i){
        rChannels[i] = input_jpeg.buffer[i*input_jpeg.num_channels];
        gChannels[i] = input_jpeg.buffer[i*input_jpeg.num_channels+1];
        bChannels[i] = input_jpeg.buffer[i*input_jpeg.num_channels+2];
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    int chunk_size = (input_jpeg.width-2) * (input_jpeg.height-2) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        // TODO
        thread_data[i].rChannels = rChannels;  
        thread_data[i].gChannels = gChannels;  
        thread_data[i].bChannels = bChannels;  
        thread_data[i].output_buffer = filterImage;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? (input_jpeg.width-2) * (input_jpeg.height-2) : (i + 1) * chunk_size;
        
        pthread_create(&threads[i], nullptr, conv_func, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Write GrayImage to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filterImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filterImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
