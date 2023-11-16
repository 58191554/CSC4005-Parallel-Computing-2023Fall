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

typedef std::vector<int> vi;



void merge(std::vector<int>& vec, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary vectors
    std::vector<int> L(n1);
    std::vector<int> R(n2);

    // Copy data to temporary vectors L[] and R[]
    for (int i = 0; i < n1; i++) {
        L[i] = vec[l + i];
    }
    for (int i = 0; i < n2; i++) {
        R[i] = vec[m + 1 + i];
    }

    // Merge the temporary vectors back into v[l..r]
    int i = 0; // Initial index of the first subarray
    int j = 0; // Initial index of the second subarray
    int k = l; // Initial index of the merged subarray

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            vec[k] = L[i];
            i++;
        } else {
            vec[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        vec[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        vec[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& vec, int num_threads) {
    vi cuts(num_threads+1, 0);
    int N = vec.size();
    int num_per_thread = N/num_threads;
    int left_num = N%num_threads;
    int divide_left_num = 0;
    for(int i = 0; i < num_threads; i++){
        if(divide_left_num<left_num){
            cuts[i+1] = cuts[i] + num_per_thread + 1;
            divide_left_num++;
        }else{
            cuts[i+1] = cuts[i] + num_per_thread;
        }
    }
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i){
        std::sort(vec.begin() + cuts[i], vec.begin() + cuts[i+1]);
    }

    // merge
    int num_cuts = num_threads;
    while(num_cuts>1){
        #pragma omp parallel for 
        for(int i = 0; i< num_cuts-1; i+=2){
            merge(vec, cuts[i], cuts[i+1]-1, cuts[i+2]-1);
        }
        int tmp_cuts = (num_cuts%2==0) ? num_cuts/2 : num_cuts/2 +1;
        // std::cout << "new cuts: ";
        for(int i = 0; i < num_cuts; i++){
            if(i < tmp_cuts){
                cuts[i+1] = cuts[2*(i+1)];
            }
            else{
                cuts[i+1] = 0;
            }
            // std::cout << cuts[i] << ", ";
        }
        // std::cout << std::endl;
        num_cuts = tmp_cuts;
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

    mergeSort(vec, thread_num);
    // for(int i = 0; i < size; ++i){
    //     std::cout << vec[i] << ", ";
    // }
    // std::cout << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);
}