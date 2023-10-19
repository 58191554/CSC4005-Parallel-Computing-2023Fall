//
// Created by Zhen TONG on 2023/9/30.
// Email: 120090694@link.cuhk.edu.cn
//
// OpenMP implementation of smooth filtering
//

#include <iostream>
#include <chrono>
#include <omp.h>    // OpenMP header
#include <vector>
#include "utils.hpp"
#define PAD 8

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    auto filterImage = new unsigned char [(input_jpeg.width-2) * (input_jpeg.height-2) * input_jpeg.num_channels];
    auto rchannel = new unsigned char[(input_jpeg.width) * (input_jpeg.height)];
    auto gchannel = new unsigned char[(input_jpeg.width) * (input_jpeg.height)];
    auto bchannel = new unsigned char[(input_jpeg.width) * (input_jpeg.height)];
    for(int i = 0; i < input_jpeg.width * input_jpeg.height; i++){
        rchannel[i] = input_jpeg.buffer[i*input_jpeg.num_channels];
        gchannel[i] = input_jpeg.buffer[i*input_jpeg.num_channels+1];
        bchannel[i] = input_jpeg.buffer[i*input_jpeg.num_channels+2];
    }

    // Divide the tasks
    int num_threads = atoi(argv[3]); 
    omp_set_num_threads(num_threads);
    int total_pixel_num = (input_jpeg.width-2) * (input_jpeg.height-2);
    int pixel_num_per_task = total_pixel_num / num_threads;
    int left_pixel_num = total_pixel_num % num_threads;

    std::vector<int> cuts(num_threads + 1, 0);
    int divided_left_pixel_num = 0;

    for (int i = 0; i < num_threads; ++i) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            divided_left_pixel_num++;
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }

    // Start 
    auto start_time = std::chrono::high_resolution_clock::now();
    int num_threads_check = 0;
    #pragma omp parallel default(none) shared(rchannel, gchannel, bchannel, filterImage, input_jpeg, cuts, num_threads, num_threads_check)
    {
        if (omp_get_thread_num() == 0){
            num_threads_check = omp_get_num_threads();
        }
        #pragma omp for
        for(int id = 0; id < num_threads; id++){
            for(int i = cuts[id]; i < cuts[id+1]; i++){
                int height = i/(input_jpeg.width-2)+1;
                int width = i-(height-1)*(input_jpeg.width-2)+1;
                int kernel[9] = {
                    (height-1)*input_jpeg.width+width-1,
                    (height-1)*input_jpeg.width+width,
                    (height-1)*input_jpeg.width+width+1,
                    height*input_jpeg.width+width-1,
                    height*input_jpeg.width+width,
                    height*input_jpeg.width+width+1,
                    (height+1)*input_jpeg.width+width-1,
                    (height+1)*input_jpeg.width+width,
                    (height+1)*input_jpeg.width+width+1
                };
                unsigned char r = 0;
                unsigned char g = 0;
                unsigned char b = 0;
                for(int k = 0; k < 9; k++){
                    r += rchannel[kernel[k]]/9;
                    g += gchannel[kernel[k]+1]/9;
                    b += bchannel[kernel[k]+2]/9;
                }
                filterImage[i*input_jpeg.num_channels]   = static_cast<unsigned char>(r);
                filterImage[i*input_jpeg.num_channels+1] = static_cast<unsigned char>(g);
                filterImage[i*input_jpeg.num_channels+2] = static_cast<unsigned char>(b);
            }
        }
    }    
    std::cout << "Thread number check:" << num_threads_check << std::endl;
    
    // End 
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filterImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }

    // Release Heap Recourse
    delete[] input_jpeg.buffer;
    delete[] filterImage;
    delete[] rchannel;
    delete[] gchannel;
    delete[] bchannel;


    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;

}