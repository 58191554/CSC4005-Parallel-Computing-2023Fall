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

int binarySearch(vi &vec, int val) {
    int low = 0;
    int high = vec.size();

    while (low < high) {
        int mid = low + (high - low) / 2;

        if (vec[mid]<= val) {
            low = mid+1;
        } else {
            high = mid;
        }
    }

    return low; // Returns the rightmost position
}

void merge(std::vector<int>& vec, int l, int m, int r, int num_thread) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary vectors
    std::vector<int> L(n1);
    std::vector<int> R(n2);
    #pragma omp parallel for
    // Copy data to temporary vectors L[] and R[]
    for (int i = 0; i < n1; i++) {
        L[i] = vec[l + i];
    }
    #pragma omp parallel for
    for (int i = 0; i < n2; i++) {
        R[i] = vec[m + 1 + i];
    }

    int num_used = vec.size()/(r-l);
    int num_chunck = (num_thread-num_used)/num_used;
    num_chunck = num_chunck>2 ? num_chunck : 2;
    int k = n1/num_chunck;

    vi cuts_L(num_chunck+1, 0);
    for(int i = 1; i < num_chunck; i++){
        cuts_L[i] = i*k;
    }
    cuts_L[num_chunck] = n1;
    vi cuts_R(num_chunck+1, 0);
    vi cuts_LR(num_chunck+1, 0);
    for(int i = 1; i < num_chunck; i++){
        cuts_R[i] = binarySearch(R, L[cuts_L[i]]);
        cuts_LR[i] = cuts_R[i] + cuts_L[i];
    }
    cuts_R[num_chunck] = n2;
    cuts_LR[num_chunck] = n1+n2;
    #pragma omp parallel for
    for(int t = 0; t < num_chunck; t++){
        int i = cuts_L[t]; // Initial index of the first subarray
        int j = cuts_R[t]; // Initial index of the second subarray
        int k = cuts_LR[t]; // Initial index of the merged subarray

        while (i < cuts_L[t+1] && j < cuts_R[t+1]) {
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
        while (i < cuts_L[t+1]) {
            vec[k] = L[i];
            i++;
            k++;
        }

        // Copy the remaining elements of R[], if there are any
        while (j < cuts_R[t+1]) {
            vec[k] = R[j];
            j++;
            k++;
        }
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
            merge(vec, cuts[i], cuts[i+1]-1, cuts[i+2]-1, num_threads);
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