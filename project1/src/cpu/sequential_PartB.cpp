//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// A naive sequential implementation of image filtering
//

#include <iostream>
#include <cmath>
#include <chrono>

#include "utils.hpp"

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    // Apply the filter to the image
    auto filteredImage = new unsigned char[(input_jpeg.width-2) * (input_jpeg.height-2) * input_jpeg.num_channels];
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height+16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height+16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height+16];

    // load data to channel in heap
    for (int i = 0; i < (input_jpeg.width-2) * (input_jpeg.height-2); i++ ){
        filteredImage[i*input_jpeg.num_channels] = 0; filteredImage[i*input_jpeg.num_channels+1] = 0; filteredImage[i*input_jpeg.num_channels+2] = 0;
        reds[i] = input_jpeg.buffer[i*input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i*input_jpeg.num_channels+1];
        blues[i] = input_jpeg.buffer[i*input_jpeg.num_channels+2];
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    // Nested for loop, please optimize it
    for (int i = 1; i < input_jpeg.height - 1; i++)
    {
        for (int j = 1; j < input_jpeg.width - 1; j++)
        {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            int indices[9] = {
                (i - 1) * input_jpeg.width + j - 1,
                (i - 1) * input_jpeg.width + j,
                (i - 1) * input_jpeg.width + j + 1,
                i * input_jpeg.width + j - 1,
                i * input_jpeg.width + j,
                i * input_jpeg.width + j + 1,
                (i + 1) * input_jpeg.width + j - 1,
                (i + 1) * input_jpeg.width + j,
                (i + 1) * input_jpeg.width + j + 1
            };
            for(int k = 0; k < 9; k++){
                sum_r+=reds[indices[k]]*1/9;
                sum_g+=greens[indices[k]]*1/9;
                sum_b+=blues[indices[k]]*1/9;
            }
            
            int buffer_index = ((i-1)*(input_jpeg.width-2)+j-2) * input_jpeg.num_channels;
            filteredImage[buffer_index] = static_cast<unsigned char>(std::round(sum_r));
            filteredImage[buffer_index + 1] = static_cast<unsigned char>(std::round(sum_g));
            filteredImage[buffer_index + 2] = static_cast<unsigned char>(std::round(sum_b));
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] reds;
    delete[] greens;
    delete[] blues;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
