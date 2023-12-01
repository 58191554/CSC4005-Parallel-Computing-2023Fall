# Project 4: Parallel Programming with Machine Learning

```cpp
    // Allocate memory for logits and gradients arrays
    float *logits = new float[m * k]();
    float *gradients = new float[n * k]();

    // Loop over minibatches
    for (size_t start = 0; start < m; start += batch) {
        size_t end = std::min(start + batch, m);

        // Calculate logits for each example
        for (size_t i = start; i < end; i++) {
            for (size_t j = 0; j < k; j++) {
                for (size_t l = 0; l < n; l++) {
                    logits[i * k + j] += X[i * n + l] * theta[l * k + j];
                }
            }
        }

        // Apply softmax function to logits
        for (size_t i = start; i < end; i++) {
            float max_logit = logits[i * k];
            for (size_t j = 1; j < k; j++) {
                max_logit = std::max(max_logit, logits[i * k + j]);
            }

            float sum_exp_logit = 0.0;
            for (size_t j = 0; j < k; j++) {
                logits[i * k + j] = std::exp(logits[i * k + j] - max_logit);
                sum_exp_logit += logits[i * k + j];
            }

            for (size_t j = 0; j < k; j++) {
                logits[i * k + j] /= sum_exp_logit;
            }
        }

        // Calculate gradients and accumulate over minibatch
        for (size_t i = start; i < end; i++) {
            for (size_t j = 0; j < k; j++) {
                gradients[j] += (logits[i * k + j] - (y[i] == j)) * X[i * n];
            }

            for (size_t l = 1; l < n; l++) {
                for (size_t j = 0; j < k; j++) {
                    gradients[l * k + j] += (logits[i * k + j] - (y[i] == j)) * X[i * n + l];
                }
            }
        }

        // Update theta values
        for (size_t l = 0; l < n; l++) {
            for (size_t j = 0; j < k; j++) {
                theta[l * k + j] -= (lr / batch) * gradients[l * k + j];
            }
        }

        // Reset gradients for next minibatch
        std::fill(gradients, gradients + n * k, 0.0);
    }

    // Deallocate memory for logits and gradients arrays
    delete[] logits;
    delete[] gradients;

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
| Epoch | Train Loss | Train Err | Test Loss | Test Err |
|     0 |    2.29209 |   0.89367 |   2.29511 |  0.89320 |
|     1 |    2.29235 |   0.89432 |   2.29529 |  0.89360 |
|     2 |    2.29311 |   0.89417 |   2.29598 |  0.89370 |
|     3 |    2.29360 |   0.89383 |   2.29637 |  0.89360 |
|     4 |    2.29386 |   0.89362 |   2.29655 |  0.89320 |
|     5 |    2.29400 |   0.89338 |   2.29662 |  0.89310 |
|     6 |    2.29410 |   0.89335 |   2.29664 |  0.89300 |
|     7 |    2.29417 |   0.89308 |   2.29665 |  0.89270 |
|     8 |    2.29424 |   0.89288 |   2.29665 |  0.89250 |
|     9 |    2.29430 |   0.89278 |   2.29667 |  0.89260 |
Execution Time: 58362 milliseconds

```

## Reference

https://github.com/tonyyxliu/CSC4005-2023Fall/tree/main/project4

https://github.com/hanquanjushi/10-714/blob/main/hw0/src/simple_ml_ext.cpp