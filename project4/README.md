# Project 4: Parallel Programming with Machine Learning

```cpp
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
}
```



```cpp
    float * Z_b = new float[m*n];
    float * Y = new float[m*k];
    int input_dim = n;
    int num_classes = k;
    int num_img = m;
    float * gradients = new float[input_dim*num_classes];
    // BEGIN YOUR CODE
    for(int i = 0; i < m; i+=batch){
        auto X_b = X + i*input_dim;
        matrix_dot(X_b, theta, Z_b, batch, input_dim, num_classes);
        matrix_softmax_normalize(Z_b, batch, num_classes);
        vector_to_one_hot_matrix(y, Y, batch, num_classes);
        matrix_minus(Z_b, Y, batch, num_classes);
        // matrix_dot_trans(X_b, Z_b, gradients, batch, input_dim, num_classes);
        // Calculate gradients and accumulate over minibatch
        for (size_t x = i; x < i+batch; x++) {
            for (size_t j = 0; j < k; j++) {
                gradients[j] += (Z_b[x * k + j] - (y[x] == j)) * X[x * n];
            }

            for (size_t l = 1; l < n; l++) {
                for (size_t j = 0; j < k; j++) {
                    gradients[l * k + j] += (Z_b[x * k + j] - (y[x] == j)) * X[x * n + l];
                }
            }
        }
        float alpha = lr/batch;
        matrix_mul_scalar(gradients, alpha, input_dim, num_classes);
        matrix_minus(theta, gradients, input_dim, num_classes);
    }
    delete[] Z_b;
    delete[] Y;
    delete[] gradients;

```

```cpp
        matrix_softmax_normalize(local_logit, length, k);
        vector_to_one_hot_matrix(y, local_Y, length, k);
        matrix_minus(local_logit, local_Y, length, k);
        matrix_dot_trans(X + start*n, local_logit, gradients, length, n, k);        
        float alpha = lr/length;
        matrix_mul_scalar(gradients, alpha, n, k);
        matrix_minus(theta, gradients, n, k);
        std::fill(gradients, gradients + n * k, 0.0);

```



```
Training softmax regression
input_dim = 784
| Epoch | Train Loss | Train Err | Test Loss | Test Err |
|     0 |    2.67117 |   0.10182 |   2.66763 |  0.09400 |
|     1 |    2.62657 |   0.09448 |   2.62391 |  0.08830 |
|     2 |    2.61604 |   0.09002 |   2.61380 |  0.08600 |
|     3 |    2.61150 |   0.08712 |   2.60972 |  0.08480 |
|     4 |    2.60885 |   0.08478 |   2.60746 |  0.08350 |
|     5 |    2.60708 |   0.08300 |   2.60597 |  0.08170 |
|     6 |    2.60577 |   0.08152 |   2.60479 |  0.08090 |
|     7 |    2.60479 |   0.08072 |   2.60386 |  0.08080 |
|     8 |    2.60400 |   0.07972 |   2.60304 |  0.08060 |
|     9 |    2.60337 |   0.07907 |   2.60235 |  0.08020 |
Execution Time: 46108 milliseconds

```

## Reference

https://github.com/tonyyxliu/CSC4005-2023Fall/tree/main/project4

https://github.com/hanquanjushi/10-714/blob/main/hw0/src/simple_ml_ext.cpp