//
// Created by Zhen TONG on 2023/9/30.
// Email: 120090694@link.cuhk.edu.cn
//
// CUDA implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>

#include <cuda_runtime.h> // CUDA Header

#include "utils.hpp"

// CUDA kernel functon：convolution 3x3 kernel
__global__ void conv_func(const unsigned char* input, unsigned char* output, int width, int height, int num_channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height){
        // TODO multiply pixel with 1/9, and sum up the 3x3 pixels into output
        // The Data in input is like:[pixel0_r, pixel0_g, pixel0_b, pixel1_r, pixel_r, pixel_b, ...]
        // The output is like:[pixel0_r, pixel0_g, pixel0_b, pixel1_r, pixel_r, pixel_b, ...] but in size (width-2)x(height-2)
        int row = idx / (width - 2); // Calculate the row of the output pixel
        int col = idx % (width - 2); // Calculate the column of the output pixel

        int output_index = row * (width - 2) + col; // Calculate the index for the output pixel

        // Initialize the sum for the convolution
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;

        // Apply the 3x3 convolution kernel
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int input_row = row + i;
                int input_col = col + j;
                int input_index = (input_row * width + input_col) * num_channels;

                // Extract the RGB values from the input
                unsigned char r = input[input_index];
                unsigned char g = input[input_index + 1];
                unsigned char b = input[input_index + 2];

                // Apply the convolution kernel weights and accumulate the sum
                sum_r += static_cast<float>(r) / 9.0f;
                sum_g += static_cast<float>(g) / 9.0f;
                sum_b += static_cast<float>(b) / 9.0f;
            }
        }

        // Store the result in the output array
        output[output_index * num_channels] = static_cast<unsigned char>(sum_r);
        output[output_index * num_channels + 1] = static_cast<unsigned char>(sum_g);
        output[output_index * num_channels + 2] = static_cast<unsigned char>(sum_b);        
    }
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Allocate memory on host (CPU)
    auto filterImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];   
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, 
                input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_output,
               (input_jpeg.width-2) * (input_jpeg.height-2) * input_jpeg.num_channels * sizeof(unsigned char));
    // Copy input data from host to device
    cudaMemcpy(d_input, 
                input_jpeg.buffer, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char),
                cudaMemcpyHostToDevice);
    // Computation: Convultion
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blockSize = 512; // 256
    // int numBlocks =
    //     (input_jpeg.width * input_jpeg.height + blockSize - 1) / blockSize;
    int numBlocks = ((input_jpeg.width-2) * (input_jpeg.height-2)) / blockSize + 1;
    cudaEventRecord(start, 0); // GPU start time
    conv_func<<<numBlocks, blockSize>>>(d_input, d_output, input_jpeg.width,
                                        input_jpeg.height,
                                        input_jpeg.num_channels);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(filterImage, d_output,
               (input_jpeg.width-2) * (input_jpeg.height-2) * input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    // Write GrayImage to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filterImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};

    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] filterImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}