//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort
//

#include <iostream>
#include <vector>
#include "../utils.hpp"
#define THRESHOLD 1000

void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quickSortParallel(int arr[], int low, int high, int threads) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Check if the task is large enough to parallelize
        if (high - low > THRESHOLD) {
#pragma omp parallel sections
            {
#pragma omp section
                {
                    // Recursive call to the left partition
                    #pragma omp task
                    quickSortParallel(arr, low, pi - 1, threads);
                }
#pragma omp section
                {
                    // Recursive call to the right partition
                    #pragma omp task
                    quickSortParallel(arr, pi + 1, high, threads);
                }
            }
        } else {
            // If the task is small, do it sequentially
            quickSortParallel(arr, low, pi - 1, threads);
            quickSortParallel(arr, pi + 1, high, threads);
        }
    }
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Recursive call to the left partition
        quickSort(arr, low, pi - 1);

        // Recursive call to the right partition
        quickSort(arr, pi + 1, high);
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable threads_num vector_size\n"
            );
    }

    const int thread_num = atoi(argv[1]);

    const int size = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp single nowait
        {    
            quickSortParallel(&vec[0], 0, size - 1, thread_num);
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Quick Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);

    return 0;
}