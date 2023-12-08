#include "simple_ml_openacc.hpp"

int main()
{
    DataSet *train_data = parse_mnist("./dataset/training/train-images.idx3-ubyte",
                                      "./dataset/training/train-labels.idx1-ubyte");
    DataSet *test_data = parse_mnist("./dataset/testing/t10k-images.idx3-ubyte",
                                     "./dataset/testing/t10k-labels.idx1-ubyte");

    std::cout << "Training softmax regression (GPU)" << std::endl;
    train_softmax_openacc(train_data, test_data, 10, 10, 0.2);
    delete train_data;
    delete test_data;


    // float * A = new float[3*2];
    // float * B = new float[3*2];
    // float * C = new float[2*2];
    // float data = 0.0f;
    // for(size_t i = 0; i < 6; i++){
    //     A[i] = data;
    //     data += 1;
    // }
    // for(size_t i = 0; i < 6; i++){
    //     B[i] = data;
    //     data += 1;
    // }
    // #pragma acc enter data copyin(A[0:6], B[0:6], C[0:4])
    // matrix_dot_trans_openacc(A, B, C, 3, 2, 2);
    // #pragma acc enter data copyin(A[0:6], B[0:6])
    // matrix_minus_openacc(A, B, 3, 2);
    // #pragma acc exit data copyout(A[0:6])
    // for(size_t i = 0; i < 6; i++){
    //     printf("%f ", A[i]);
    // }

    return 0;
}
