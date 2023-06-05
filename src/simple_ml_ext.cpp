#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
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
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    unsigned short step;

    if (m > batch) {
        step = m / batch;
    } else {
        step = 1;
    }

    for (auto batch_index{0}; batch_index < step; batch_index++) {
        float batch_X[batch * n];
        float batch_X_T[batch * n];
        float batch_y[batch];
        float exp[batch * k];
        float Z[batch * k];
        float Z_ey[batch * k];
        float row_sum_exp[batch];
        float ey[batch * k];
        float grad[n*k];

        // Get the batch_X
        for (long unsigned int i{0}; i < batch; i++) {
            for (long unsigned int j{0}; j < n; j++) {
                batch_X[i * n + j] = X[(i + (batch_index * batch)) * n + j];
            }
        }

        // Get the batch_y
        for (long unsigned int i{0}; i < batch; i++) {
            batch_y[i] = y[i + (batch_index * batch)];
        }

        // Calculate np.exp(np.dot(batch_X, theta) as exp
        for (long unsigned int col{0}; col < k; col++){
            for (long unsigned int i{0}; i < batch; i++) {
                float cell{0};
                for (long unsigned int j{0}; j < n; j++) {
                    auto dot_product = batch_X[i * n + j] * theta[j * k + col];
                    cell = cell + dot_product;
                }
                exp[i * k + col] = std::exp(cell);
            }
        }

        // Calculate np.sum(np.exp(np.dot(batch_X, theta)) as row_sum_exp
        for (long unsigned int i{0}; i < batch; i++) {
            row_sum_exp[i] = 0.0f;
            for (long unsigned int j{0}; j < k; j++) {
                row_sum_exp[i] += exp[i * k + j];
            }
        }

        // Calculate Z
        for (long unsigned int i{0}; i < batch; i++) {
            for (long unsigned int j{0}; j < k; j++) {
                Z[i * k + j] = exp[i * k + j] / row_sum_exp[i];
            }
        }

        // Calculate identity ey
        for (long unsigned int i{0}; i < batch; i++) {
            for (long unsigned int j{0}; j < k; j++) {
                if (j == batch_y[i]) {
                    ey[i * k + j] = 1;
                } else {
                    ey[i * k + j] = 0;
                }
            }
        }

        // Calculate Z - ey
        for (long unsigned int i{0}; i < batch; i++) {
            for (long unsigned int j{0}; j < k; j++) {
                Z_ey[i * k + j] = Z[i * k + j] - ey[i * k + j];
            }
        }

        // Transpose batch_X
        for (long unsigned int j{0}; j < n; j++) {
            for (long unsigned int i{0}; i < batch; i++) {
                batch_X_T[j*batch + i] = batch_X[i*n + j];
            }
        }

        // Calculate grad = (1/batch) * np.dot(batch_X_T, (Z - ey))
        for (long unsigned int col{0}; col < k; col++){
            for (long unsigned int i{0}; i < n; i++) {
                float cell{0};
                for (long unsigned int j{0}; j < batch; j++) {
                    auto dot_product = batch_X_T[i * batch + j] * Z_ey[j * k + col];
                    cell += dot_product;
                }
                grad[i * k + col] = (1/float(batch)) * cell;
            }
        }

        // Update the theta
        for (long unsigned int i{0}; i < n; i++){
            for (long unsigned int j{0}; j < k; j++){
                theta[i* k + j] -= lr * grad[i* k + j];
            }
        }

    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
