# SIMD

## V1

```cpp
//
// Created by Zhen Tong on 2023/9/28.
// Email: 120090694@link.cuhk.edu.cn
//
// SIMD (AVX2) implementation of Convolution
//

#include <iostream>
#include <chrono>
#include <cmath>

#include <immintrin.h>

#include "utils.hpp"
int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Transform the RGB Contents to Filtered
    auto filteredImage = new unsigned char[(input_jpeg.width-2) * (input_jpeg.height-2)*input_jpeg.num_channels + 8];

    // Prepross, store reds, greens and blues separately
    auto reds = new unsigned char[32];
    auto greens = new unsigned char[32];
    auto blues = new unsigned char[32];

    for (int i = 0; i < 32; i++){
        reds[i] = 0;
        greens[i] = 0;
        blues[i] = 0;
    }

    for(int i = 1; i < input_jpeg.height-1; i++){
        for(int j = 1; j < input_jpeg.width-1; j++){
            filteredImage[((i-1)*(input_jpeg.width-2)+j-1)*input_jpeg.num_channels] = 1/9*input_jpeg.buffer[(i*input_jpeg.width+j)*input_jpeg.num_channels];
            filteredImage[((i-1)*(input_jpeg.width-2)+j-1)*input_jpeg.num_channels+1] = 1/9*input_jpeg.buffer[(i*input_jpeg.width+j)*input_jpeg.num_channels+1];
            filteredImage[((i-1)*(input_jpeg.width-2)+j-1)*input_jpeg.num_channels+2] = 1/9*input_jpeg.buffer[(i*input_jpeg.width+j)*input_jpeg.num_channels+2];
        }
    }


    // set the SIMD weight in the fileter
    __m256 one_ninth_256 = _mm256_set1_ps(1.0f / 9.0f);


    auto start_time = std::chrono::high_resolution_clock::now();
    // Use SIMD

    for (int i = 1; i < input_jpeg.height-1; i++){
        for (int j = 1; j < input_jpeg.width-1; j++){
            int indices[8] = {
                (i - 1) * input_jpeg.width + j - 1,
                (i - 1) * input_jpeg.width + j,
                (i - 1) * input_jpeg.width + j + 1,
                i * input_jpeg.width + j - 1,
                // i * input_jpeg.width + j,
                i * input_jpeg.width + j + 1,
                (i + 1) * input_jpeg.width + j - 1,
                (i + 1) * input_jpeg.width + j,
                (i + 1) * input_jpeg.width + j + 1
            };

            // prepare 9 square index into a linear group
            for(int l = 0; l < 8; l++){
                reds[l] = input_jpeg.buffer[indices[l] * input_jpeg.num_channels];
                greens[l] = input_jpeg.buffer[indices[l] * input_jpeg.num_channels+1];
                blues[l] = input_jpeg.buffer[indices[l] * input_jpeg.num_channels + 2];
            }
            // Load the 8 red chars to a 256 bits float register

            __m128i red_chars = _mm_loadu_si128((__m128i*) (reds));
            __m256i red_ints = _mm256_cvtepu8_epi32(red_chars);
            __m256 red_floats = _mm256_cvtepi32_ps(red_ints);
            __m256 red_results = _mm256_mul_ps(red_floats, one_ninth_256);
            __m128 red_sum_128 = _mm_add_ps(_mm256_extractf128_ps(red_results, 0), _mm256_extractf128_ps(red_results, 1));
            __m128 red_sum = _mm_hadd_ps(red_sum_128, red_sum_128);
            float red_result_float = _mm_cvtss_f32(red_sum);
            int red_result_int = static_cast<int>(red_result_float)*2;
            filteredImage[((i-1)*(input_jpeg.width-2)+(j-1))*input_jpeg.num_channels] += static_cast<unsigned int>(red_result_int);

            __m128i green_chars = _mm_loadu_si128((__m128i*) (greens));
            __m256i green_ints = _mm256_cvtepu8_epi32(green_chars);
            __m256 green_floats = _mm256_cvtepi32_ps(green_ints);
            __m256 green_results = _mm256_mul_ps(green_floats, one_ninth_256);
            __m128 green_sum_128 = _mm_add_ps(_mm256_extractf128_ps(green_results, 0), _mm256_extractf128_ps(green_results, 1));
            __m128 green_sum = _mm_hadd_ps(green_sum_128, green_sum_128);
            float green_result_float = _mm_cvtss_f32(green_sum);
            int green_result_int = static_cast<int>(green_result_float)*2;
            filteredImage[((i-1)*(input_jpeg.width-2)+(j-1))*input_jpeg.num_channels+1] += static_cast<unsigned int>(green_result_int);

            __m128i blue_chars = _mm_loadu_si128((__m128i*) (blues));
            __m256i blue_ints = _mm256_cvtepu8_epi32(blue_chars);
            __m256 blue_floats = _mm256_cvtepi32_ps(blue_ints);
            __m256 blue_results = _mm256_mul_ps(blue_floats, one_ninth_256);
            __m128 blue_sum_128 = _mm_add_ps(_mm256_extractf128_ps(blue_results, 0), _mm256_extractf128_ps(blue_results, 1));
            __m128 blue_sum = _mm_hadd_ps(blue_sum_128, blue_sum_128);
            float blue_result_float = _mm_cvtss_f32(blue_sum);
            int blue_result_int = static_cast<int>(blue_result_float)*2;
            filteredImage[((i-1)*(input_jpeg.width-2)+(j-1))*input_jpeg.num_channels+2] += static_cast<unsigned int>(blue_result_int);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
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
    delete[] blues;
    delete[] greens;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;

}
```



## V2

```cpp
            __m128i color_chars1 = _mm_loadu_si128((__m128i*)(reds+left_up));
            __m128i color_chars2 = _mm_loadu_si128((__m128i*)(reds+left_up+1));
            __m128i color_chars3 = _mm_loadu_si128((__m128i*)(reds+left_up+2));
            __m128i color_chars4 = _mm_loadu_si128((__m128i*)(reds+left_mid));
            __m128i color_chars5 = _mm_loadu_si128((__m128i*)(reds+left_mid+1));
            __m128i color_chars6 = _mm_loadu_si128((__m128i*)(reds+left_mid+2));
            __m128i color_chars7 = _mm_loadu_si128((__m128i*)(reds+left_down));
            __m128i color_chars8 = _mm_loadu_si128((__m128i*)(reds+left_down+1));
            __m128i color_chars9 = _mm_loadu_si128((__m128i*)(reds+left_down+2));

            __m256i color_ints1 = _mm256_cvtepu8_epi32(color_chars1);
            __m256i color_ints2 = _mm256_cvtepu8_epi32(color_chars2);
            __m256i color_ints3 = _mm256_cvtepu8_epi32(color_chars3);
            __m256i color_ints4 = _mm256_cvtepu8_epi32(color_chars4);
            __m256i color_ints5 = _mm256_cvtepu8_epi32(color_chars5);
            __m256i color_ints6 = _mm256_cvtepu8_epi32(color_chars6);
            __m256i color_ints7 = _mm256_cvtepu8_epi32(color_chars7);
            __m256i color_ints8 = _mm256_cvtepu8_epi32(color_chars8);
            __m256i color_ints9 = _mm256_cvtepu8_epi32(color_chars9);

            __m256 red_floats = _mm256_cvtepi32_ps(color_ints1);
            __m256 red_floats = _mm256_cvtepi32_ps(color_ints2);
            __m256 red_floats = _mm256_cvtepi32_ps(color_ints3);
            __m256 red_floats = _mm256_cvtepi32_ps(color_ints4);
            __m256 red_floats = _mm256_cvtepi32_ps(color_ints5);
            __m256 red_floats = _mm256_cvtepi32_ps(color_ints6);
            __m256 red_floats = _mm256_cvtepi32_ps(color_ints7);
            __m256 red_floats = _mm256_cvtepi32_ps(color_ints8);
            __m256 red_floats = _mm256_cvtepi32_ps(color_ints9);

```



# MPI

## V1

```cpp
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

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            divided_left_pixel_num++;
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }


    auto start_time = std::chrono::high_resolution_clock::now();
    if(taskid == MASTER){
        // master process
        auto filterImage = new unsigned char[input_jpeg.width*input_jpeg.height*input_jpeg.num_channels];

        // master do the kernel work for the first cut
        for(int i = cuts[MASTER]; i < cuts[MASTER+1]; i++){
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
                r += input_jpeg.buffer[kernel[k]*input_jpeg.num_channels]/9;
                g += input_jpeg.buffer[kernel[k]*input_jpeg.num_channels+1]/9;
                b += input_jpeg.buffer[kernel[k]*input_jpeg.num_channels+2]/9;
            }
            filterImage[i*input_jpeg.num_channels] = r;
            filterImage[i*input_jpeg.num_channels+1] = g;
            filterImage[i*input_jpeg.num_channels+2] = b;
        }
        
        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filterImage + cuts[i]*input_jpeg.num_channels;
            int length = (cuts[i+1] - cuts[i])*input_jpeg.num_channels;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
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
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }else{
        // slave process
        int cut_len = cuts[taskid+1]-cuts[taskid];
        auto filterImageSeg = new unsigned char[cut_len*input_jpeg.num_channels];
        for(int i = cuts[taskid]; i < cuts[taskid+1]; i++){
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
                r += input_jpeg.buffer[kernel[k]*input_jpeg.num_channels]/9;
                g += input_jpeg.buffer[kernel[k]*input_jpeg.num_channels+1]/9;
                b += input_jpeg.buffer[kernel[k]*input_jpeg.num_channels+2]/9;
            }
            filterImageSeg[(i-cuts[taskid])*input_jpeg.num_channels] = r;
            filterImageSeg[(i-cuts[taskid])*input_jpeg.num_channels+1] = g;
            filterImageSeg[(i-cuts[taskid])*input_jpeg.num_channels+2] = b;
        }


        // Send the filter segmentation to the master process
        MPI_Send(filterImageSeg, cut_len*input_jpeg.num_channels,  MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        // Release heap source
        delete[] filterImageSeg;
    }
    return 0;
}
```

## **V2**

```cpp
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

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            divided_left_pixel_num++;
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }


    auto start_time = std::chrono::high_resolution_clock::now();
    if(taskid == MASTER){
        // master process
        auto filterImage = new unsigned char[input_jpeg.width*input_jpeg.height*input_jpeg.num_channels];
        auto reds = new unsigned char[cuts[taskid+1]-cuts[taskid]];
        auto greens = new unsigned char[cuts[taskid+1]-cuts[taskid]];
        auto blues = new unsigned char[cuts[taskid+1]-cuts[taskid]];

        for(int i = cuts[taskid]; i < cuts[taskid+1]; i++){
            reds[i-cuts[taskid]] = input_jpeg.buffer[i*input_jpeg.num_channels];
            greens[i-cuts[taskid]] = input_jpeg.buffer[i*input_jpeg.num_channels+1];
            blues[i-cuts[taskid]] = input_jpeg.buffer[i*input_jpeg.num_channels+2];
        }

        // master do the kernel work for the first cut
        for(int i = cuts[MASTER]; i < cuts[MASTER+1]; i++){
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
                r += reds[kernel[k]-cuts[taskid]]/9;
                g += greens[kernel[k]-cuts[taskid]]/9;
                b += blues[kernel[k]-cuts[taskid]]/9;
            }
            filterImage[i*input_jpeg.num_channels] = r;
            filterImage[i*input_jpeg.num_channels+1] = g;
            filterImage[i*input_jpeg.num_channels+2] = b;
        }
        
        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filterImage + cuts[i]*input_jpeg.num_channels;
            int length = (cuts[i+1] - cuts[i])*input_jpeg.num_channels;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
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
        delete[] reds;
        delete[] greens;
        delete[] blues;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }else{
        // slave process
        int cut_len = cuts[taskid+1]-cuts[taskid];
        auto filterImageSeg = new unsigned char[cut_len*input_jpeg.num_channels];
        auto reds = new unsigned char[cuts[taskid+1]-cuts[taskid]];
        auto greens = new unsigned char[cuts[taskid+1]-cuts[taskid]];
        auto blues = new unsigned char[cuts[taskid+1]-cuts[taskid]];

        for(int i = cuts[taskid]; i < cuts[taskid+1]; i++){
            reds[i-cuts[taskid]] = input_jpeg.buffer[i*input_jpeg.num_channels];
            greens[i-cuts[taskid]] = input_jpeg.buffer[i*input_jpeg.num_channels+1];
            blues[i-cuts[taskid]] = input_jpeg.buffer[i*input_jpeg.num_channels+2];
        }

        for(int i = cuts[taskid]; i < cuts[taskid+1]; i++){
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
                r += reds[kernel[k]-cuts[taskid]]/9;
                g += greens[kernel[k]-cuts[taskid]]/9;
                b += blues[kernel[k]-cuts[taskid]]/9;
            }
            filterImageSeg[(i-cuts[taskid])*input_jpeg.num_channels] = r;
            filterImageSeg[(i-cuts[taskid])*input_jpeg.num_channels+1] = g;
            filterImageSeg[(i-cuts[taskid])*input_jpeg.num_channels+2] = b;
        }


        // Send the filter segmentation to the master process
        MPI_Send(filterImageSeg, cut_len*input_jpeg.num_channels,  MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        // Release heap source
        delete[] filterImageSeg;
        delete[] reds;
        delete[] greens;
        delete[] blues;
    }
    return 0;
}
```

## V3(work)

```cpp
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

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            divided_left_pixel_num++;
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }


    auto start_time = std::chrono::high_resolution_clock::now();
    if(taskid == MASTER){
        // master process
        auto filterImage = new unsigned char[input_jpeg.width*input_jpeg.height*input_jpeg.num_channels];

        // master do the kernel work for the first cut
        for(int i = 1; i < input_jpeg.height-1; i++){
            for(int j = 2; j < input_jpeg.width-1; j++){
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
                    for(int k = 0; k < 9; k++){
                        filterImage[cuts_index*input_jpeg.num_channels  ] += input_jpeg.buffer[indices[k]*input_jpeg.num_channels]/9;
                        filterImage[cuts_index*input_jpeg.num_channels+1] += input_jpeg.buffer[indices[k]*input_jpeg.num_channels+1]/9;
                        filterImage[cuts_index*input_jpeg.num_channels+2] += input_jpeg.buffer[indices[k]*input_jpeg.num_channels+2]/9;
                    }
                }
            }
        }
        
        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filterImage + cuts[i]*input_jpeg.num_channels;
            int length = (cuts[i+1] - cuts[i])*input_jpeg.num_channels;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
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
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }else{
        // slave process
        int cut_len = cuts[taskid+1]-cuts[taskid];
        auto filterImageSeg = new unsigned char[cut_len*input_jpeg.num_channels];
        for(int i = 1; i < input_jpeg.height-1; i++){
            for(int j = 2; j < input_jpeg.width-1; j++){
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
                    for(int k = 0; k < 9; k++){
                        filterImageSeg[(cuts_index-cuts[taskid])*input_jpeg.num_channels  ] += input_jpeg.buffer[indices[k]*input_jpeg.num_channels]/9;
                        filterImageSeg[(cuts_index-cuts[taskid])*input_jpeg.num_channels+1] += input_jpeg.buffer[indices[k]*input_jpeg.num_channels+1]/9;
                        filterImageSeg[(cuts_index-cuts[taskid])*input_jpeg.num_channels+2] += input_jpeg.buffer[indices[k]*input_jpeg.num_channels+2]/9;
                    }
                }
            }
        }


        // Send the filter segmentation to the master process
        MPI_Send(filterImageSeg, cut_len*input_jpeg.num_channels,  MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        // Release heap source
        delete[] filterImageSeg;
    }
    return 0;
}
```

## V4-now

```cpp
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
        for(int i = start_row; i < input_jpeg.height-1; ++i){
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
        auto son_start_time = std::chrono::high_resolution_clock::now();
        int cut_len = cuts[taskid+1]-cuts[taskid];
        auto filterImageSeg = new unsigned char[cut_len*input_jpeg.num_channels];
        int start_row = cuts[taskid]/(input_jpeg.width-2);
        int end_row = (cuts[taskid+1]-1)/(input_jpeg.width-2)+1;
        for(int i = start_row; i < input_jpeg.height-1; ++i){
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
        auto son_end_time = std::chrono::high_resolution_clock::now();
        auto son_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(son_end_time - start_time);
        std::cout << "Son Complete!" << std::endl;
        std::cout << "Execution Time: " << son_elapsed_time.count() << " milliseconds\n";
        // Release heap source
        delete[] filterImageSeg;
    }
    return 0;
}
```



## Table

|      | MPI-V1 | MPI-V2 | Target |
| ---- | ------ | ------ | ------ |
| 1    | 9502   | 9780   | 7324   |
| 2    | 7631   | 7859   | 7134   |
| 4    | 4098   | 4263   | 3764   |
| 8    | 2350   | 2362   | 2093   |
| 16   | 1414   | 1451   | 1083   |
| 32   | 1009   | 1038   | 694    |

# OpenMP

```
//V1
//
// Created by Zhen TONG on 2023/9/30.
// Email: 120090694@link.cuhk.edu.cn
//
// OpenMP implementation of smooth filtering
//

#include <iostream>
#include <chrono>
#include <omp.h>    // OpenMP header
#include "utils.hpp"

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
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
    auto reds = new unsigned char[(input_jpeg.width-2) * (input_jpeg.height-2)];
    auto greens = new unsigned char[(input_jpeg.width-2) * (input_jpeg.height-2)];
    auto blues = new unsigned char[(input_jpeg.width-2) * (input_jpeg.height-2)];
    // Start 
    auto start_time = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for default(none) shared(reds, greens, blues, filterImage, input_jpeg)
    for(int i = 0; i < (input_jpeg.width-2) * (input_jpeg.height-2); i++){
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
            r += input_jpeg.buffer[kernel[k]*input_jpeg.num_channels]/9;
            g += input_jpeg.buffer[kernel[k]*input_jpeg.num_channels+1]/9;
            b += input_jpeg.buffer[kernel[k]*input_jpeg.num_channels+2]/9;
        }
        filterImage[i*input_jpeg.num_channels] = r;
        filterImage[i*input_jpeg.num_channels+1] = g;
        filterImage[i*input_jpeg.num_channels+2] = b;
    }

    // Save output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filterImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }
    // End 
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Release Heap Recourse
    delete[] filterImage;
    delete[] reds;
    delete[] greens;
    delete[] blues;


    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;

}
```

//
// Created by Zhen TONG on 2023/9/30.
// Email: 120090694@link.cuhk.edu.cn
//
// OpenMP implementation of smooth filtering
//

#include <iostream>
#include <chrono>
#include <omp.h>    // OpenMP header
#include "utils.hpp"

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
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

​    auto filterImage = new unsigned char [(input_jpeg.width-2) * (input_jpeg.height-2) * input_jpeg.num_channels];
​    auto reds = new unsigned char[(input_jpeg.width) * (input_jpeg.height)];
​    auto greens = new unsigned char[(input_jpeg.width) * (input_jpeg.height)];
​    auto blues = new unsigned char[(input_jpeg.width) * (input_jpeg.height)];
​    for(int i = 0; i < input_jpeg.width * input_jpeg.height; i++){
​        reds[i] = input_jpeg.buffer[i*input_jpeg.num_channels];
​        greens[i] = input_jpeg.buffer[i*input_jpeg.num_channels+1];
​        blues[i] = input_jpeg.buffer[i*input_jpeg.num_channels+2];
​    }
​    // Start 
​    auto start_time = std::chrono::high_resolution_clock::now();
​    
​    #pragma omp parallel for default(none) shared(reds, greens, blues, filterImage, input_jpeg)
​    for (int i = 1; i < input_jpeg.height-1; i++){
​        for(int j = 1; j < input_jpeg.width-1; j++){
​            unsigned char r = 0;
​            unsigned char g = 0;
​            unsigned char b = 0;
​            int indices[9] = {
​                (i - 1) * input_jpeg.width + j - 1,
​                (i - 1) * input_jpeg.width + j,
​                (i - 1) * input_jpeg.width + j + 1,
​                i * input_jpeg.width + j - 1,
​                i * input_jpeg.width + j,
​                i * input_jpeg.width + j + 1,
​                (i + 1) * input_jpeg.width + j - 1,
​                (i + 1) * input_jpeg.width + j,
​                (i + 1) * input_jpeg.width + j + 1
​            };
​            for(int k = 0; k < 9; k++){
​                r+=reds[indices[k]]*1/9;
​                g+=greens[indices[k]]*1/9;
​                b+=blues[indices[k]]*1/9;
​            }
​            
​            int buffer_index = ((i-1)*(input_jpeg.width-2)+j-2) * input_jpeg.num_channels;
​            filterImage[buffer_index] = r;
​            filterImage[buffer_index + 1] = g;
​            filterImage[buffer_index + 2] = b;
​        }
​    }
​    // Save output JPEG
​    const char* output_filepath = argv[2];
​    std::cout << "Output file to: " << output_filepath << "\n";
​    JPEGMeta output_jpeg{filterImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};
​    if (write_to_jpeg(output_jpeg, output_filepath)) {
​        std::cerr << "Failed to save output JPEG image\n";
​        return -1;
​    }
​    // End 
​    auto end_time = std::chrono::high_resolution_clock::now();
​    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

​    // Release Heap Recourse
​    delete[] filterImage;
​    delete[] reds;
​    delete[] greens;
​    delete[] blues;


​    std::cout << "Transformation Complete!" << std::endl;
​    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
​    return 0;

## V2

```cpp
//
// Created by Zhen TONG on 2023/9/30.
// Email: 120090694@link.cuhk.edu.cn
//
// OpenMP implementation of smooth filtering
//

#include <iostream>
#include <chrono>
#include <omp.h>    // OpenMP header
#include "utils.hpp"

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
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
    auto reds = new unsigned char[(input_jpeg.width) * (input_jpeg.height)];
    auto greens = new unsigned char[(input_jpeg.width) * (input_jpeg.height)];
    auto blues = new unsigned char[(input_jpeg.width) * (input_jpeg.height)];
    for(int i = 0; i < input_jpeg.width * input_jpeg.height; i++){
        reds[i] = input_jpeg.buffer[i*input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i*input_jpeg.num_channels+1];
        blues[i] = input_jpeg.buffer[i*input_jpeg.num_channels+2];
    }
    // Start 
    auto start_time = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for default(none) shared(reds, greens, blues, filterImage, input_jpeg)
    for (int i = 1; i < input_jpeg.height-1; i++){
        for(int j = 1; j < input_jpeg.width-1; j++){
            unsigned char r = 0;
            unsigned char g = 0;
            unsigned char b = 0;
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
                r+=reds[indices[k]]*1/9;
                g+=greens[indices[k]]*1/9;
                b+=blues[indices[k]]*1/9;
            }
            
            int buffer_index = ((i-1)*(input_jpeg.width-2)+j-2) * input_jpeg.num_channels;
            filterImage[buffer_index] = r;
            filterImage[buffer_index + 1] = g;
            filterImage[buffer_index + 2] = b;
        }
    }
    // Save output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filterImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }
    // End 
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Release Heap Recourse
    delete[] filterImage;
    delete[] reds;
    delete[] greens;
    delete[] blues;


    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;

}
```

## V3(betterbut not best)

```cpp
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
                input_jpeg.buffer[i*input_jpeg.num_channels]   = static_cast<unsigned char>(r);
                input_jpeg.buffer[i*input_jpeg.num_channels+1] = static_cast<unsigned char>(g);
                input_jpeg.buffer[i*input_jpeg.num_channels+2] = static_cast<unsigned char>(b);
            }
        }
    }    
    std::cout << "Thread number check:" << num_threads_check << std::endl;
    
    // Save output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filterImage, input_jpeg.width-2, input_jpeg.height-2, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }
    // End 
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

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
```



## Table

|      | V2   | target |
| ---- | ---- | ------ |
| 1    | 9280 | 8542   |
| 2    | 9236 | 7299   |
| 4    | 6899 | 3886   |
| 8    | 5201 | 1862   |
| 16   | 4153 | 1089   |
| 32   | 3525 | 605    |

# pthread

## V1(good but not best)

```cpp
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
    unsigned char* input_buffer;
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
    for (int i = data->start; i < data->end; i++) {
        int height = i/(data->width-2)+1;
        int width = i-(height-1)*(data->width-2)+1;
        int kernel[9] = {
            (height-1)*data->width+width-1,
            (height-1)*data->width+width,
            (height-1)*data->width+width+1,
            height*data->width+width-1,
            height*data->width+width,
            height*data->width+width+1,
            (height+1)*data->width+width-1,
            (height+1)*data->width+width,
            (height+1)*data->width+width+1
        };
        unsigned char r = 0;
        unsigned char g = 0;
        unsigned char b = 0;
        for(int k = 0; k < 9; k++){
            r += data->input_buffer[kernel[k]*data->num_channels]/9;
            g += data->input_buffer[kernel[k]*data->num_channels+1]/9;
            b += data->input_buffer[kernel[k]*data->num_channels+2]/9;
        }
        data->output_buffer[i*data->num_channels]   = static_cast<unsigned char>(r);
        data->output_buffer[i*data->num_channels+1] = static_cast<unsigned char>(g);
        data->output_buffer[i*data->num_channels+2] = static_cast<unsigned char>(b);
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

    auto start_time = std::chrono::high_resolution_clock::now();

    int chunk_size = (input_jpeg.width-2) * (input_jpeg.height-2) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        // TODO
        thread_data[i].input_buffer = input_jpeg.buffer;  //It is necessry to load all the image into the shit?
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

```

## V2-(better but not best)

```cpp
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
    for (int i = data->start; i < data->end; i++) {
        int height = i/(data->width-2)+1;
        int width = i-(height-1)*(data->width-2)+1;
        int kernel[9] = {
            (height-1)*data->width+width-1,
            (height-1)*data->width+width,
            (height-1)*data->width+width+1,
            height*data->width+width-1,
            height*data->width+width,
            height*data->width+width+1,
            (height+1)*data->width+width-1,
            (height+1)*data->width+width,
            (height+1)*data->width+width+1
        };
        unsigned char r = 0;
        unsigned char g = 0;
        unsigned char b = 0;
        for(int k = 0; k < 9; k++){
            r += data->rChannels[kernel[k]*data->num_channels]/9;
            g += data->gChannels[kernel[k]*data->num_channels+1]/9;
            b += data->bChannels[kernel[k]*data->num_channels+2]/9;
        }
        data->output_buffer[i*data->num_channels]   = static_cast<unsigned char>(r);
        data->output_buffer[i*data->num_channels+1] = static_cast<unsigned char>(g);
        data->output_buffer[i*data->num_channels+2] = static_cast<unsigned char>(b);
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

```

## V3(best)-now

```cpp
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

```









## Table

|      | V1   | Target |
| ---- | ---- | ------ |
| 1    | 9707 | 8066   |
| 2    | 7446 | 7229   |
| 4    | 3958 | 3836   |
| 8    | 2138 | 1835   |
| 16   | 946  | 924    |
| 32   | 526  | 535    |

# CUDA

## V1

```cpp
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
    // TODO
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height)
    {
            int h = idx/(width-2)+1;
            int w = idx-(height-1)*(width-2)+1;
            int kernel[9] = {
                (h-1)*width+w-1,
                (h-1)*width+w,
                (h-1)*width+w+1,
                h*width+w-1,
                h*width+w,
                h*width+w+1,
                (h+1)*width+w-1,
                (h+1)*width+w,
                (h+1)*width+w+1
            };
            // unsigned char r = input[kernel[0]*num_channels]/9;
            // r+=input[kernel[1]*num_channels]/9;
            // r+=input[kernel[2]*num_channels]/9;
            // r+=input[kernel[3]*num_channels]/9;
            // r+=input[kernel[4]*num_channels]/9;
            // r+=input[kernel[5]*num_channels]/9;
            // r+=input[kernel[6]*num_channels]/9;
            // r+=input[kernel[7]*num_channels]/9;
            // r+=input[kernel[8]*num_channels]/9;

            // unsigned char g = input[kernel[0]*num_channels+1]/9;
            // g+=input[kernel[1]*num_channels+1]/9;
            // g+=input[kernel[2]*num_channels+1]/9;
            // g+=input[kernel[3]*num_channels+1]/9;
            // g+=input[kernel[4]*num_channels+1]/9;
            // g+=input[kernel[5]*num_channels+1]/9;
            // g+=input[kernel[6]*num_channels+1]/9;
            // g+=input[kernel[7]*num_channels+1]/9;
            // g+=input[kernel[8]*num_channels+1]/9;

            // unsigned char b = input[kernel[0]*num_channels+2]/9;
            // b+=input[kernel[1]*num_channels+2]/9;
            // b+=input[kernel[2]*num_channels+2]/9;
            // b+=input[kernel[3]*num_channels+2]/9;
            // b+=input[kernel[4]*num_channels+2]/9;
            // b+=input[kernel[5]*num_channels+2]/9;
            // b+=input[kernel[6]*num_channels+2]/9;
            // b+=input[kernel[7]*num_channels+2]/9;
            // b+=input[kernel[8]*num_channels+2]/9;
            unsigned char r = input[idx*num_channels];
            unsigned char g = input[idx*num_channels+1];
            unsigned char b = input[idx*num_channels+2];
            

            output[idx*num_channels] = static_cast<unsigned char>(r);
            output[idx*num_channels+1] = static_cast<unsigned char>(g);
            output[idx*num_channels+2] = static_cast<unsigned char>(b);
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
```

## V2

```cpp
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
```

|      | Target | V2      |
| ---- | ------ | ------- |
| 1    | 32     | 39.6089 |
|      |        |         |
|      |        |         |

# OpenACC

## V1



# DashTable

| My   | Sequential                      | SIMD (AVX2)                     | MPI                             | Pthread | OpenMP                          | CUDA    | OpenACC                       |
| ---- | ------------------------------- | ------------------------------- | ------------------------------- | ------- | ------------------------------- | ------- | ----------------------------- |
| 1    | <font color = green>6661</font> | <font color = green>2878</font> | <font color = green>6156</font> | 6779    | <font color = black>8784</font> | 39.6351 | <font color = green>20</font> |
| 2    | N/A                             | N/A                             | <font color = green>5922</font> | 5897    | <font color = green>6729</font> |         |                               |
| 4    | N/A                             | N/A                             | <font color = black>3349</font> | 3288    | <font color = green>3716</font> |         |                               |
| 8    | N/A                             | N/A                             | <font color = green>1689</font> | 1653    | <font color = green>1879</font> |         |                               |
| 16   | N/A                             | N/A                             | <font color = black>1102</font> | 938     | <font color = green>1098</font> |         |                               |
| 32   | N/A                             | N/A                             | <font color = black>716</font>  | 536     | <font color = green>661</font>  |         |                               |

| TAs  | Sequential | SIMD (AVX2) | MPI  | Pthread | OpenMP | CUDA | OpenACC |
| ---- | ---------- | ----------- | ---- | ------- | ------ | ---- | ------- |
| 1    | 7247       | 4335        | 7324 | 8066    | 8542   | 32   | 23      |
| 2    | N/A        | N/A         | 7134 | 7229    | 7299   | N/A  | N/A     |
| 4    | N/A        | N/A         | 3764 | 3836    | 3886   | N/A  | N/A     |
| 8    | N/A        | N/A         | 2093 | 1835    | 1862   | N/A  | N/A     |
| 16   | N/A        | N/A         | 1083 | 924     | 1089   | N/A  | N/A     |
| 32   | N/A        | N/A         | 694  | 535     | 605    | N/A  | N/A     |









