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
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (col < width && row < height)
    {
        unsigned char sum_r = 0;
        unsigned char sum_g = 0;
        unsigned char sum_b = 0;
        
        int kernel[9] = {
            (row-1)*width+col-1,
            (row-1)*width+col,
            (row-1)*width+col+1,
            row*width+col-1,
            row*width+col,
            row*width+col+1,
            (row+1)*width+col-1,
            (row+1)*width+col,
            (row+1)*width+col+1
        };
        
        for (int i = 0; i < 9; i++)
        {
                    // Extract the RGB values from the input
                    unsigned char r = input[kernel[i]*num_channels];
                    unsigned char g = input[kernel[i]*num_channels + 1];
                    unsigned char b = input[kernel[i]*num_channels + 2];
                    
                    // Apply the convolution kernel weights and accumulate the sum
                    sum_r += r / 9; // Adjust the weight as needed
                    sum_g += g / 9; // Adjust the weight as needed
                    sum_b += b / 9; // Adjust the weight as needed
            
        }
                
        // Calculate the output index for the current pixel
        int output_index = ((row-1) * (width-2) + (col-1)) * num_channels;
        
        // Store the result in the output array for all channels
        output[output_index]     = sum_r;
        output[output_index + 1] = sum_g;
        output[output_index + 2] = sum_b;
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
    int deviceID = 0; // 设备ID，如果有多个GPU，可以选择不同的设备
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    std::cout << "Max block size for device " << deviceID << ": " << deviceProp.maxThreadsPerBlock << std::endl;

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
    // int blockSize = 512; // 256
    // int numBlocks =
    //     (input_jpeg.width * input_jpeg.height + blockSize - 1) / blockSize;
    // int numBlocks = ((input_jpeg.width-2) * (input_jpeg.height-2)*input_jpeg.num_channels) / blockSize + 1;
    cudaEventRecord(start, 0); // GPU start time

    dim3 blockDim(32, 32); // You can adjust these values based on your GPU's capabilities
    dim3 gridDim((input_jpeg.width-2 + blockDim.x - 1) / blockDim.x, (input_jpeg.height-2 + blockDim.y - 1) / blockDim.y);
    
    conv_func<<<gridDim, blockDim>>>(d_input, d_output, input_jpeg.width,
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