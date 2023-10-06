//
// Created by Zhen TONG on 2023/9/30.
// Email: 120090694@link.cuhk.edu.cn
//
// OpenACC implementation of smooth filter convolution
//

#include <iostream>
#include <chrono>

#include "utils.hpp"
// #include <openacc.h> // OpenACC Header

int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char *input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JPEGMeta input_jpeg = read_from_jpeg(input_filepath);
    // Computation: Smooth Convolution
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    unsigned char *filterImage = new unsigned char[(width-2) * (height-2) * num_channels];
    unsigned char *buffer = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }
#pragma acc enter data copyin(filterImage[0 : (width-2) * (height-2) *num_channels], buffer[0 : width * height * num_channels])

#pragma acc update device(filterImage[0 : (width-2) * (height-2) *num_channels], buffer[0 : width * height * num_channels])

    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel present(filterImage[0 : (width-2) * (height-2) *num_channels], buffer[0 : width * height * num_channels]) num_gangs(1024)
    {
#pragma acc loop independent
        for (int i = 0; i < (width - 2) * (height - 2); i++)
        {
            // Initialize the sums for RGB channels
            unsigned char sum_r = 0;
            unsigned char sum_g = 0;
            unsigned char sum_b = 0;
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

            for(int k = 0; k < 9; ++k){
                sum_r += (buffer[kernel[k]*num_channels]) / 9;
                sum_g += (buffer[kernel[k]*num_channels + 1]) / 9;
                sum_b += (buffer[kernel[k]*num_channels + 2]) / 9;
            }

            // Store the results in the filterImage array
            filterImage[i * num_channels] = sum_r;
            filterImage[i * num_channels + 1] = sum_g;
            filterImage[i * num_channels + 2] = sum_b;
        }    
    }
    auto end_time = std::chrono::high_resolution_clock::now();
#pragma acc update self(filterImage[0 : (width-2) * (height-2) *num_channels], buffer[0 : width * height * num_channels])

#pragma acc exit data copyout(filterImage[0 : (width-2) * (height-2) *num_channels])

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Write GrayImage to output JPEG
    const char *output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filterImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filterImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
