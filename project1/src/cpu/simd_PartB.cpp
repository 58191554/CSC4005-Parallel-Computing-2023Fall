//
// Created by Zhen TONG on 2023/9/30.
// Email: 120090694@link.cuhk.edu.cm
//
// SIMD (AVX2) implementation of transferring a JPEG picture from RGB to gray
//

#include <iostream>
#include <chrono>

#include <immintrin.h>

#include "utils.hpp"

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Transform the RGB Contents to the gray contents
    auto filterImage = new unsigned char[(input_jpeg.width-2) * (input_jpeg.height-2)*input_jpeg.num_channels+32];

    // Prepross, store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto bracket = new unsigned char[8];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Set SIMD scalars, we use AVX2 instructions
    __m256 one_ninth = _mm256_set1_ps(1.0f/9.0f);

    // Mask used for shuffling when store int32s to u_int8 arrays
    // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
    __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1);

    // Using SIMD to accelerate the transformation
    auto start_time = std::chrono::high_resolution_clock::now();    // Start recording time

    for(int i = 0; i < input_jpeg.height-2; i++){
        // Load a 3x8x(3x3) data: data range 3x8, kernel = 3x3
        for(int j = 0; j < input_jpeg.width; j+=8){
            int left_up = i*input_jpeg.width+j;
            int left_mid = (i+1)*input_jpeg.width+j;
            int left_down = (i+2)*input_jpeg.width+j;
            int indices[9] = {
                left_up, left_up+1, left_up+2,
                left_mid, left_mid+1, left_mid+2,
                left_down, left_down+1, left_down+2
            };
            // loop for for 3 Channels RGB
            for(int c = 0; c < 3; c++){
                __m256 color_convs_floats_8 = _mm256_set1_ps(0.0f);
                __m128i color_chars_8; 

                for(int k = 0; k < 9; k++){
                    switch (c)
                    {
                    case 0:
                        color_chars_8 = _mm_loadu_si128((__m128i*) (reds+indices[k])); 
                        break;
                    case 1:
                        color_chars_8 = _mm_loadu_si128((__m128i*) (greens+indices[k])); 
                        break;
                    case 2:
                        color_chars_8 = _mm_loadu_si128((__m128i*) (blues+indices[k])); 
                        break;                    
                    }
                    __m256i color_ints_8 = _mm256_cvtepu8_epi32(color_chars_8);
                    __m256 color_floats_8 = _mm256_cvtepi32_ps(color_ints_8);
                    __m256 color_rusults_8 = _mm256_mul_ps(color_floats_8, one_ninth);
                    color_convs_floats_8 = _mm256_add_ps(color_convs_floats_8, color_rusults_8);
                }
                // Convert the float32 results to int32
                __m256i color_convs_ints_8 =  _mm256_cvtps_epi32(color_convs_floats_8);
                // Seperate the 256bits result to 2 128bits result
                __m128i low = _mm256_castsi256_si128(color_convs_ints_8);
                __m128i high = _mm256_extracti128_si256(color_convs_ints_8, 1);

                // shuffling int32s to u_int8s
                // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
                __m128i trans_low = _mm_shuffle_epi8(low, shuffle);
                __m128i trans_high = _mm_shuffle_epi8(high, shuffle);
        
                // Store the results back to gray image
                int index = i*(input_jpeg.width-2)+j;
                
                // _mm_storeu_si128((__m128i*)(&filterImage[index*input_jpeg.num_channels+c]), trans_low);
                // _mm_storeu_si128((__m128i*)(&filterImage[(index+4)*input_jpeg.num_channels+c]), trans_high);
                _mm_storeu_si128((__m128i*)(&bracket[0]), trans_low);
                _mm_storeu_si128((__m128i*)(&bracket[4]), trans_high);
                for(int l = 0;l < 8; l++){
                    filterImage[(index+l)*input_jpeg.num_channels+c] = bracket[l];
                }
            }
        }
    }



    auto end_time = std::chrono::high_resolution_clock::now();  // Stop recording time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output Gray JPEG Image
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
    delete[] reds;
    delete[] greens;
    delete[] blues;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}

