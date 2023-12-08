#include "simple_ml_openacc.hpp"

void matrix_set_zero(float *A, size_t m, size_t n){
    #pragma acc data present(A[0:m*n])
    #pragma acc parallel loop independent
    for(size_t i = 0; i < m*n; i++){
        A[i] = 0.0f;
    }
}

void matrix_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    matrix_set_zero(C, m, k);
    #pragma acc data present(A[0 :m*n], B[0 :n*k], C[0 :m*k])
    #pragma acc parallel loop collapse(2)
    for(size_t i = 0; i < m; i++){
        for(size_t l = 0; l < k; l++){
            size_t j = 0; 
            float sum = 0.0f;
            #pragma acc loop reduction(+: sum)
            for(; j < n; j ++){
                sum += A[i*n+j] * B[j*k+l];
            }
            C[i*k+l] = sum;
        }
    }
    // END YOUR CODE
}

void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
{
    // BEGIN YOUR CODE
    matrix_set_zero(C, m, k);
    #pragma acc data present(A[0:n*m], B[0:n*k], C[0:m*k])
    #pragma acc parallel loop collapse(2)
    for(size_t i = 0; i < m; i++){
        for(size_t l = 0; l < k; l++){
            float sum = 0.0f;
            #pragma acc loop reduction(+:sum)
            for(size_t j = 0; j < n; j++){
                sum += A[j*m+i] * B[j*k+l];
            }
            C[i*k+l] = sum;
        }
    }
   // END YOUR CODE
}

void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    matrix_set_zero(C, m, k);
    #pragma acc data present(A[0:m*n], B[0:k*n], C[0:m*k])
    #pragma acc parallel loop collapse(2)
    for(size_t i = 0; i < m; i++){
        for(size_t l = 0; l < k; l++){
            float sum = 0.0f;
            #pragma acc loop reduction(+:sum)
            for(size_t j = 0; j < n; j++){
                sum += A[i*n+j] * B[l*n+j];
            }
            C[i*k+l] = sum;
        }
    }
    // END YOUR CODE
}

void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc data present(A[0:m*n], B[0:m*n])
    #pragma acc parallel loop present(A[0:m*n], B[0:m*n])
    for(size_t i = 0; i < m*n; i++){
        A[i] = A[i] - B[i];
    }
}


void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop present(C[0:m*n])
    for(size_t i = 0; i < m*n; i++){
        C[i] = C[i] * scalar;
    }
    // END YOUR CODE
}

void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc data present(C[0:m*n])
    #pragma acc parallel loop independent
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

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for(size_t i = 0; i < m; i++){
        auto Y_row = Y + i*n;
        Y_row[y[i]] = 1;
    }
    // END YOUR CODE
}


void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch, float * logits,
                                      float * gradients, float * Y)
{
    // BEGIN YOUR CODE
    for(size_t start = 0; start < m-batch; start += batch){
        size_t end = std::min(start + batch, m);
        auto local_logit = logits + start*k;
        auto local_X = X + start*n;
        auto local_Y = Y + start*k;
        size_t length = end-start;
        matrix_dot_openacc(local_X, theta, local_logit, length, n, k);
        matrix_softmax_normalize_openacc(local_logit, length, k);
        matrix_minus_openacc(local_logit, local_Y, length, k);
        matrix_dot_trans_openacc(local_X, local_logit, gradients, length, n, k);
        matrix_mul_scalar_openacc(gradients, lr/length, n, k);
        matrix_minus_openacc(theta, gradients, n, k);
    }
    // END YOUR CODE
}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
    size_t m_tr = train_data->images_num;
    size_t m_te = test_data->images_num;
    size_t n = train_data->input_dim;
    auto start_time = std::chrono::high_resolution_clock::now();

    float *logits = new float[m_tr * num_classes]();
    float *gradients = new float[n * num_classes]();
    float *Y = new float[m_tr * num_classes]();
    memset(Y, 0, m_tr*num_classes*sizeof(float));
    vector_to_one_hot_matrix_openacc(train_data->labels_array, Y, m_tr, num_classes);

    #pragma acc enter data copyin(\
        theta[0:size],\
        train_result[0:size_tr],\
        test_result[0:size_te],\
        train_data->images_matrix[0:m_tr*n],\
        train_data->labels_array[0:m_tr],\
        test_data->images_matrix[0:m_te*n],\
        test_data->labels_array[0:m_te])
    #pragma acc enter data copyin(\
        logits[0:m_tr*num_classes], \
        gradients[0:n*num_classes], \
        Y[0:m_tr*num_classes])

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {   
        softmax_regression_epoch_openacc(train_data->images_matrix, train_data->labels_array, 
                                            theta, m_tr, n, num_classes, lr, batch,
                                            logits, gradients, Y);
        
        matrix_dot_openacc(train_data->images_matrix, theta, train_result, m_tr, n, num_classes);
        matrix_dot_openacc(test_data->images_matrix, theta, test_result, m_te, n, num_classes);
        
        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    
   // END YOUR CODE
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float total_loss = 0.0;
    #pragma acc data present(result[0:images_num*num_classes], labels_array[0:images_num])
    {
    #pragma acc parallel loop independent 
    for (size_t i = 0; i < images_num; ++i)
    {
        // Get the logits for the current example
        const float *logits = result + i * num_classes;

        // Get the true label for the current example
        unsigned char true_label = labels_array[i];

        // Compute the softmax probabilities and the loss
        float sum_exp = 0.0;
        float loss = 0.0;

        // Compute the sum of exponentials (without subtracting the max_logit)
        for (size_t j = 0; j < num_classes; ++j)
        {
            float exp_logit = expf(logits[j]);
            sum_exp += exp_logit;
        }

        // Compute the softmax probabilities and the loss
        for (size_t j = 0; j < num_classes; ++j)
        {
            float softmax_prob = expf(logits[j]) / sum_exp;

            if (j == true_label)
            {
                loss = -logf(softmax_prob);
            }
        }

        // Accumulate the loss for the current example
        total_loss += loss;
    }
    }
    // Compute the average loss over all examples
    float average_loss = total_loss / images_num;
    // END YOUR CODE
    return average_loss;
}

float mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    #pragma acc data present(result[0:images_num*num_classes], labels_array[0:images_num])
    float probability;
    int output;
    float err = 0;
    #pragma acc parallel loop independent
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

void matrix_mul_openacc(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE
    #pragma acc data present(A[0:size], B[0:size])
    #pragma acc parallel loop independent
    for(size_t i = 0; i < size; i++){
        A[i] *= B[i];
    }
    // END YOUR CODE
}

void matrix_relu_openacc(float *A, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc data present(A[0:m*n])
    #pragma acc parallel loop collapse(2)
    for(size_t i = 0; i < m; i++){
        for(size_t j = 0; j < n; j++){
            A[i*n+j] = A[i*n+j] > 0 ? A[i*n+j] : 0;
        }
    }
    // END YOUR CODE
}

void matrix_relu_cache_openacc(float * A, float * A_cache, size_t m, size_t n){
    #pragma acc data present(A[0:m*n], A_cache[0:m*n])
    #pragma acc parallel loop independent
    for(size_t i = 0; i < m; i++){
        auto A_row = A + i*n;
        auto A_cacheRow = A_cache + i*n;
        for(size_t j = 0; j < n; j++){
            A_row[j] = A_row[j] > 0 ? A_row[j] : 0;
            A_cacheRow[j] = A_row[j] > 0 ? 1 : 0;
        }
    }
}

void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE
    float *Z1 = new float[batch * l]();
    float *Z1_cache = new float[batch * l]();
    float *Z2 = new float[batch * k]();
    float *Y = new float[m*k]();
    memset(Y, 0, m*k*sizeof(float));
    vector_to_one_hot_matrix(y, Y, m, k);
    float *G1 = new float[batch * l]();
    float *W1_l = new float[n * l]();
    float *W2_l = new float[l * k]();
    #pragma acc enter data copyin(\
        Z1[0:batch*l],\
        Z1_cache[0:batch*l],\
        Z2[0:batch*k],\
        Y[0:m*k],\
        G1[0:batch*l],\
        W1_l[0:n*l],\
        W2_l[0:l*k])
    #pragma acc data present(\
        X[0:m*n],\
        W1[0:n*l],\
        W2[0:l*k])
    for(size_t start = 0; start < m; start += batch){
        size_t end = std::min(start + batch, m);
        size_t length = end-start;
        auto X_b = X + start*n;
        matrix_dot_openacc(X_b, W1, Z1, length, n, l);  // l = 400, unable to directly unroll.
        matrix_relu_cache_openacc(Z1, Z1_cache, length, l);
        matrix_dot_openacc(Z1, W2, Z2, length, l, k); // k = 10 direcly unroll
        matrix_softmax_normalize_openacc(Z2, length, k);
        auto Y_b = Y + start*k;
        matrix_minus_openacc(Z2, Y_b, length, k);
        matrix_trans_dot_openacc(Z2, W2, G1, length, k, l); // k = 10 direcly unroll
        matrix_mul_openacc(G1, Z1_cache, length*l);
        matrix_dot_trans_openacc(X_b, G1, W1_l, length, n, l); // l = 400, unable to directly unroll.
        matrix_mul_scalar_openacc(W1_l, lr/batch, n, l);
        matrix_minus_openacc(W1, W1_l, n, l);
        matrix_dot_trans_openacc(Z1, Z2, W2_l, length, l, k); // k = 10 direcly unroll
        matrix_mul_scalar_openacc(W2_l, lr/batch, l, k);
        matrix_minus_openacc(W2, W2_l, l, k);

    }
    #pragma acc exit data delete(\
        Z1[0:batch*l], \
        Z1_cache[0:batch*l], \
        Z2[0:batch*k], \
        Y[0:m*k], \
        G1[0:batch*l], \
        W1_l[0:n*l], \
        W2_l[0:l*k])
    delete[] Z1;
    delete[] Z1_cache;
    delete[] Z2;
    delete[] Y;
    delete[] G1;
    delete[] W1_l;
    delete[] W2_l;
    // END YOUR CODE
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
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
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    size_t m_tr = train_data->images_num;
    size_t m_te = test_data->images_num;
    size_t n = train_data->input_dim;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  
    float *Z1_train = new float[train_data->images_num*hidden_dim];
    float *Z1_test = new float[test_data->images_num*hidden_dim];

    #pragma acc enter data copyin(\
        train_data->images_matrix[0:m_tr*n],\
        train_data->labels_array[0:m_tr],\
        test_data->images_matrix[0:m_te*n],\
        test_data->labels_array[0:m_te],\
        W1[0:size_w1],\
        W2[0:size_w2],\
        Z1_train[0:m_tr*hidden_dim],\
        Z1_test[0:m_te*hidden_dim],\
        train_result[0:m_tr*num_classes],\
        test_result[0:m_te*num_classes])

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        nn_epoch_openacc(train_data->images_matrix, train_data->labels_array, W1, W2, \
            train_data->images_num, train_data->input_dim, hidden_dim, num_classes, \
            lr, batch);
        matrix_dot_openacc(train_data->images_matrix, W1, Z1_train, train_data->images_num,\
            train_data->input_dim, hidden_dim);
        matrix_dot_openacc(test_data->images_matrix, W1, Z1_test, test_data->images_num,\
            test_data->input_dim, hidden_dim);
        // // ReLU
        matrix_relu_openacc(Z1_train, train_data->images_num, hidden_dim);
        matrix_relu_openacc(Z1_test, test_data->images_num, hidden_dim);
        // // Z2 = dot(Z1, W2)
        matrix_dot_openacc(Z1_train, W2, train_result, train_data->images_num, hidden_dim, num_classes);
        matrix_dot_openacc(Z1_test, W2, test_result, test_data->images_num, hidden_dim, num_classes);

// THE BUG IS HERE something wrong
        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
    // END YOUR CODE
    delete[] W1;
    delete[] W2;
    delete[] Z1_train;
    delete[] train_result;
    delete[] Z1_test ;
    delete[] test_result ;
}
