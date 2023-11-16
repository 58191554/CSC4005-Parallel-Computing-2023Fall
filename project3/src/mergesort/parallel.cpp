//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Merge Sort
//

#include <iostream>
#include <vector>
#include "../utils.hpp"
#include <cmath>

void merge(std::vector<int>& vec, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> leftArray(n1);
    std::vector<int> rightArray(n2);

    for (int i = 0; i < n1; i++)
        leftArray[i] = vec[left + i];
    for (int j = 0; j < n2; j++)
        rightArray[j] = vec[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (leftArray[i] <= rightArray[j]) {
            vec[k] = leftArray[i];
            i++;
        } else {
            vec[k] = rightArray[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        vec[k] = leftArray[i];
        i++;
        k++;
    }

    while (j < n2) {
        vec[k] = rightArray[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& vec, int left, int right, int threshold) {
    if (left < right) {
        if (right - left <= threshold) {
            // Sequential merge sort for small chunks
            std::sort(vec.begin() + left, vec.begin() + right + 1);
        } else {
            int mid = left + (right - left) / 2;

            #pragma omp parallel sections
            {
                #pragma omp section
                mergeSort(vec, left, mid, threshold);

                #pragma omp section
                mergeSort(vec, mid + 1, right, threshold);
            }

            merge(vec, left, mid, right);
        }
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

    std::vector<int> S(size);
    std::vector<int> L(size);
    std::vector<int> results(size);

    auto start_time = std::chrono::high_resolution_clock::now();
    int n = 1;
    while(std::pow(2, n)<thread_num+1){
        n++;
    }
    int threshold = size/(std::pow(2, n-1))+1;

    mergeSort(vec, 0, size - 1, threshold);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);
}