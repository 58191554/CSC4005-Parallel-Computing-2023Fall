#include "simple_ml_ext.hpp"
#include <math.h>
#include <cmath>

DataSet::DataSet(size_t images_num, size_t input_dim)
    : images_num(images_num), input_dim(input_dim)
{
    images_matrix = new float[images_num * input_dim];
    labels_array = new unsigned char[images_num];
}

DataSet::~DataSet()
{
    delete[] images_matrix;
    delete[] labels_array;
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/**
 *Read an images and labels file in MNIST format.  See this page:
 *http://yann.lecun.com/exdb/mnist/ for a description of the file format.
 *Args:
 *    image_filename (str): name of images file in MNIST format (idx3-ubyte)
 *    label_filename (str): name of labels file in MNIST format (idx1-ubyte)
 **/
DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename)
{
    std::ifstream images_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream labels_file(label_filename, std::ios::in | std::ios::binary);
    uint32_t magic_num, images_num, rows_num, cols_num;

    images_file.read(reinterpret_cast<char *>(&magic_num), 4);
    labels_file.read(reinterpret_cast<char *>(&magic_num), 4);

    images_file.read(reinterpret_cast<char *>(&images_num), 4);
    labels_file.read(reinterpret_cast<char *>(&images_num), 4);
    images_num = swap_endian(images_num);

    images_file.read(reinterpret_cast<char *>(&rows_num), 4);
    rows_num = swap_endian(rows_num);
    images_file.read(reinterpret_cast<char *>(&cols_num), 4);
    cols_num = swap_endian(cols_num);

    DataSet *dataset = new DataSet(images_num, rows_num * cols_num);

    labels_file.read(reinterpret_cast<char *>(dataset->labels_array), images_num);
    unsigned char *pixels = new unsigned char[images_num * rows_num * cols_num];
    images_file.read(reinterpret_cast<char *>(pixels), images_num * rows_num * cols_num);
    for (size_t i = 0; i < images_num * rows_num * cols_num; i++)
    {
        dataset->images_matrix[i] = static_cast<float>(pixels[i]) / 255;
    }

    delete[] pixels;

    return dataset;
}

/**
 *Print Matrix
 *Print the elements of a matrix A with size m * n.
 *Args:
 *      A (float*): Matrix of size m * n
 **/
void print_matrix(float *A, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    memset(C, 0, m * k * sizeof(float));
    for(size_t i = 0; i < m; i ++){
        auto A_row = A + i*n;
        auto C_row = C + i*k;
        for(size_t l = 0; l < n; l++){
            auto B_row = B + l*k;
            float A_il = A_row[l];
            for(size_t j = 0; j < k; j++){
                C_row[j] += A_il*B_row[j];
            }
        }
    }
    // END YOUR CODE
}

void matrix_dot_unroll(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    memset(C, 0, m * k * sizeof(float));
    for(size_t i = 0; i < m; i ++){
        auto A_row = A + i*n;
        auto C_row = C + i*k;
        for(size_t l = 0; l < n; l++){
            auto B_row = B + l*k;
            float A_il = A_row[l];
            C_row[0] += A_il*B_row[0];
            C_row[1] += A_il*B_row[1];
            C_row[2] += A_il*B_row[2];
            C_row[3] += A_il*B_row[3];
            C_row[4] += A_il*B_row[4];
            C_row[5] += A_il*B_row[5];
            C_row[6] += A_il*B_row[6];
            C_row[7] += A_il*B_row[7];
            C_row[8] += A_il*B_row[8];
            C_row[9] += A_il*B_row[9];
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size n * m
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot_trans(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
{
    memset(C, 0, m * k * sizeof(float));
    int input_dim = m;
    int batch_size = n;
    int num_class = k;
    // BEGIN YOUR CODE
    for(size_t x = 0; x < n; x++){
        auto A_row = A + x*m;
        auto B_row = B + x*k;
        for(size_t i = 0; i < m; i++){
            auto C_row = C + i*k;
            for(size_t j = 0; j < k; ++j){
                C_row[j] += A_row[i] * B_row[j];
            }
        }
    }
    // END YOUR CODE
}

void matrix_dot_trans_unroll(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
{
    memset(C, 0, m * k * sizeof(float));
    int input_dim = m;
    int batch_size = n;
    int num_class = k;
    // BEGIN YOUR CODE
    for(size_t x = 0; x < n; x++){
        auto A_row = A + x*m;
        auto B_row = B + x*k;
        for(size_t i = 0; i < m; i++){
            auto C_row = C + i*k;
            float A_row_i = A_row[i];
            C_row[0] += A_row_i * B_row[0];
            C_row[1] += A_row_i * B_row[1];
            C_row[2] += A_row_i * B_row[2];
            C_row[3] += A_row_i * B_row[3];
            C_row[4] += A_row_i * B_row[4];
            C_row[5] += A_row_i * B_row[5];
            C_row[6] += A_row_i * B_row[6];
            C_row[7] += A_row_i * B_row[7];
            C_row[8] += A_row_i * B_row[8];
            C_row[9] += A_row_i * B_row[9];
        }
    }
    // END YOUR CODE
}


/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size k * n
 *     C (float*): Matrix of size m * k
 **/
void matrix_trans_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // for(int i = 0; i < m*n; i++){
    //     A[i] -= B[i];
    // }
    for(int i = 0; i < m; i++){
        auto A_row = A + i*n;
        auto B_row = B + i*n;
        for(int j = 0; j < n; ++j){
            A_row[j] -= B_row[j];
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for(size_t i = 0; i < m; i++){
        auto C_row = C + i*n;
        for(size_t j = 0; j < n; ++j){
            C_row[j] *= scalar;
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for(size_t i = 0; i < m; i++){
        auto C_row = C + i*n;
        for(size_t j = 0; j < n; ++j){
            C_row[j] /= scalar;
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // Iterate over each row
    for (size_t i = 0; i < m; ++i) {
        // Find the maximum value in the row
        auto C_row = C + i*n;
        float max_val = C_row[0];

        for (size_t j = 1; j < n; ++j) {
            if (C_row[j] > max_val) {
                max_val = C_row[j];
            }
        }

        // Apply softmax normalization to the row
        float sum_exp = 0.0;
        for (size_t j = 0; j < n; ++j) {
            // C_row[j] = expf(C_row[j] - max_val);
            C_row[j] = expf(C_row[j]);
            sum_exp += C_row[j];
        }

        // Normalize the row
        for (size_t j = 0; j < n; ++j) {
            C_row[j] /= sum_exp;
        }
    }    
    // END YOUR CODE
}

void matrix_softmax_normalize_unroll(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // Iterate over each row
    for (size_t i = 0; i < m; ++i) {
        // Find the maximum value in the row
        auto C_row = C + i*n;
        float max_val = C_row[0];

        for (size_t j = 1; j < n; ++j) {
            if (C_row[j] > max_val) {
                max_val = C_row[j];
            }
        }

        // Apply softmax normalization to the row
        float sum_exp = 0.0;
        C_row[0] = expf(C_row[0]);
        C_row[1] = expf(C_row[1]);
        C_row[2] = expf(C_row[2]);
        C_row[3] = expf(C_row[3]);
        C_row[4] = expf(C_row[4]);
        C_row[5] = expf(C_row[5]);
        C_row[6] = expf(C_row[6]);
        C_row[7] = expf(C_row[7]);
        C_row[8] = expf(C_row[8]);
        C_row[9] = expf(C_row[9]);

        sum_exp += C_row[0];
        sum_exp += C_row[1];
        sum_exp += C_row[2];
        sum_exp += C_row[3];
        sum_exp += C_row[4];
        sum_exp += C_row[5];
        sum_exp += C_row[6];
        sum_exp += C_row[7];
        sum_exp += C_row[8];
        sum_exp += C_row[9];
        
        // Normalize the row
        C_row[0] /= sum_exp;
        C_row[1] /= sum_exp;
        C_row[2] /= sum_exp;
        C_row[3] /= sum_exp;
        C_row[4] /= sum_exp;
        C_row[5] /= sum_exp;
        C_row[6] /= sum_exp;
        C_row[7] /= sum_exp;
        C_row[8] /= sum_exp;
        C_row[9] /= sum_exp;
    }    
    // END YOUR CODE
}


/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     y (unsigned char *): vector of size m * 1
 *     Y (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char *y, float *Y, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for(size_t i = 0; i < m; i++){
        auto Y_row = Y + i*n;
        Y_row[y[i]] = 1;
    }
    // END YOUR CODE
}

/**
 * A C++ version of the softmax regression epoch code.  This should run a
 * single epoch over the data defined by X and y (and sizes m,n,k), and
 * modify theta in place.  Your function will probably want to allocate
 * (and then delete) some helper arrays to store the Z_b and gradients.
 *
 * Args:
 *     X (const float *): pointer to X data, of size m*n, stored in row
 *          major (C) format
 *     y (const unsigned char *): pointer to y data, of size m
 *     theta (float *): pointer to theta data, of size n*k, stored in row
 *          major (C) format
 *     m (size_t): number of examples
 *     n (size_t): input dimension
 *     k (size_t): number of classes
 *     lr (float): learning rate / SGD step size
 *     batch (int): size of SGD batch
 *
 * Returns:
 *     (None)
 */
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch)
{
    float *logits = new float[m * k]();
    float *gradients = new float[n * k]();
    float *Y = new float[m*k]();
    vector_to_one_hot_matrix(y, Y, m, k);
    // Loop over minibatches
    for (size_t start = 0; start < m; start += batch) {
        size_t end = std::min(start + batch, m);
        auto local_logit = logits + start*k;
        auto local_X = X + start*n;
        auto local_Y = Y + start*k;
        size_t length = end-start;
        matrix_dot_unroll(local_X, theta, local_logit, length, n, k);
        // Apply softmax function to logits
        matrix_softmax_normalize_unroll(local_logit, length, k);
        // Calculate gradients and accumulate over minibatch
        matrix_minus(local_logit, local_Y, length, k);
        matrix_dot_trans_unroll(local_X, local_logit, gradients, length, n, k);
        // Update theta values
        matrix_mul_scalar(gradients, lr/length, n, k);
        // matrix_div_scalar(gradients, length, n, k);
        matrix_minus(theta, gradients, n, k);
        // Reset gradients for next minibatch
        std::fill(gradients, gradients + n * k, 0.0);
        }
    // Deallocate memory for logits and gradients arrays
    delete[] logits;
    delete[] gradients;
    delete[] Y;
}

/**
 *Example function to fully train a softmax classifier
 **/
void train_softmax(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    size_t input_dim = train_data->input_dim;
    size_t img_num = train_data->images_num;
    std::cout << "input_dim = " << input_dim << std::endl;

    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();


    for (size_t epoch = 0; epoch < epochs; epoch++)
    {   
        // BEGIN YOUR CODE
        softmax_regression_epoch_cpp(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, input_dim, num_classes, lr, batch);
        matrix_dot_unroll(train_data->images_matrix, theta, train_result, train_data->images_num, input_dim, num_classes);
        matrix_dot_unroll(test_data->images_matrix, theta, test_result, test_data->images_num, input_dim, num_classes);
        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";

    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

/*
 *Return softmax loss.  Note that for the purposes of this assignment,
 *you don't need to worry about "nicely" scaling the numerical properties
 *of the log-sum-exp computation, but can just compute this directly.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average softmax loss over the sample.
 */
float mean_softmax_loss(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    // float cross_entropy_loss = 0;
    // for(int i = 0; i < images_num; ++i){
    //     float z_y = 0;
    //     auto result_row = result + i*num_classes;
    //     for(int j = 0; j < num_classes; ++j){
    //         if(result[i*num_classes + j]>z_y){
    //             z_y = result_row[j];
    //         }
    //     }
    //     float entropy = 0;
    //     for(int j = 0; j < num_classes; j++){
    //         entropy += expf(result_row[j]);
    //     }
    //     entropy = log2f(entropy);
    //     cross_entropy_loss += (-z_y + entropy);
    // }
    // return cross_entropy_loss/images_num;
    float total_loss = 0.0;

    for (size_t i = 0; i < images_num; ++i)
    {
        // Get the logits for the current example
        const float *logits = &result[i * num_classes];

        // Get the true label for the current example
        unsigned char true_label = labels_array[i];

        // Compute the softmax probabilities and the loss
        float sum_exp = 0.0;
        float loss = 0.0;

        // Compute the sum of exponentials (without subtracting the max_logit)
        for (size_t j = 0; j < num_classes; ++j)
        {
            float exp_logit = exp(logits[j]);
            sum_exp += exp_logit;
        }

        // Compute the softmax probabilities and the loss
        for (size_t j = 0; j < num_classes; ++j)
        {
            float softmax_prob = exp(logits[j]) / sum_exp;

            if (j == true_label)
            {
                loss = -log(softmax_prob);
            }
        }

        // Accumulate the loss for the current example
        total_loss += loss;
    }

    // Compute the average loss over all examples
    float average_loss = total_loss / images_num;

    return average_loss;

    // END YOUR CODE
}

float mean_softmax_loss_unroll(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float cross_entropy_loss = 0;
    for(int i = 0; i < images_num; ++i){
        float z_y = 0;
        auto result_row = result + i*num_classes;
        for(int j = 0; j < num_classes; ++j){
            if(result[i*num_classes + j]>z_y){
                z_y = result_row[j];
            }
        }
        float entropy = 0;
        // unroll
        entropy += expf(result_row[0]);
        entropy += expf(result_row[1]);
        entropy += expf(result_row[2]);
        entropy += expf(result_row[3]);
        entropy += expf(result_row[4]);
        entropy += expf(result_row[5]);
        entropy += expf(result_row[6]);
        entropy += expf(result_row[7]);
        entropy += expf(result_row[8]);
        entropy += expf(result_row[9]);

        entropy = log2f(entropy);
        cross_entropy_loss += (-z_y + entropy);
    }
    return cross_entropy_loss/images_num;
    // END YOUR CODE
}

/*
 *Return error.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average error over the sample.
 */
float mean_err(float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float probability;
    int output;
    float err = 0;
    for(int i = 0; i < images_num; i++){
        probability = 0;
        output = 0;
        // find the max probability output
        auto result_row = result + i*num_classes;
        for(int j = 0; j < num_classes; j++){
            if(result_row[j]>probability){
                probability = result_row[j];
                output = j;
            }
        }
        // compute err
        if(output == static_cast<int>(labels_array[i])){
            err += 0;
        }else{
            err += 1;
        }
    }

    return err/images_num;
    // END YOUR CODE
}

/**
 * Matrix Multiplication
 * Efficiently compute A = A * B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_mul(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

/*
Run a single epoch of SGD for a two-layer neural network defined by the
weights W1 and W2 (with no bias terms):
    Z_b = ReLU(X * W1) * W2
The function should use the step size lr, and the specified batch size (and
again, without randomizing the order of X).  It should modify the
W1 and W2 matrices in place.
Args:
    X: 1D input array of size
        (num_examples x input_dim).
    y: 1D class label array of size (num_examples,)
    W1: 1D array of first layer weights, of shape
        (input_dim x hidden_dim)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD batch
*/
void nn_epoch_cpp(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

/**
 *Example function to fully train a nn classifier
 **/
void train_nn(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
    matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE

        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
