#include "matrix_ops.hpp"
#include <cstring>
#include <algorithm>
#include <thread>
#include <future>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

py::array_t<double> fast_matrix_multiply_cpp(py::array_t<double> A, py::array_t<double> B, bool use_parallel) {
    auto buf_A = A.request();
    auto buf_B = B.request();
    
    if (buf_A.ndim != 2 || buf_B.ndim != 2) {
        throw std::runtime_error("Input matrices must be 2-dimensional");
    }
    
    if (buf_A.shape[1] != buf_B.shape[0]) {
        throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
    }
    
    int m = buf_A.shape[0];  // rows of A
    int n = buf_B.shape[1];  // cols of B
    int k = buf_A.shape[1];  // cols of A / rows of B
    
    double* ptr_A = static_cast<double*>(buf_A.ptr);
    double* ptr_B = static_cast<double*>(buf_B.ptr);
    
    // Create output matrix
    auto result = py::array_t<double>(m * n);
    result.resize({m, n});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Initialize result matrix to zero
    std::fill(result_ptr, result_ptr + m * n, 0.0);
    
    // Optimized matrix multiplication with parallelization
    #ifdef _OPENMP
    if (use_parallel) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int l = 0; l < k; ++l) {
                    sum += ptr_A[i * k + l] * ptr_B[l * n + j];
                }
                result_ptr[i * n + j] = sum;
            }
        }
    } else
    #endif
    {
        // Sequential version with better cache locality
        for (int i = 0; i < m; ++i) {
            for (int l = 0; l < k; ++l) {
                double a_il = ptr_A[i * k + l];
                for (int j = 0; j < n; ++j) {
                    result_ptr[i * n + j] += a_il * ptr_B[l * n + j];
                }
            }
        }
    }
    
    return result;
}

py::array_t<double> optimized_indicator_matrix_cpp(py::array_t<int> sectors, py::array_t<int> unique_sectors) {
    auto sectors_buf = sectors.request();
    auto unique_buf = unique_sectors.request();
    
    if (sectors_buf.ndim != 1 || unique_buf.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }
    
    int n_firms = sectors_buf.shape[0];
    int n_sectors = unique_buf.shape[0];
    
    int* sectors_ptr = static_cast<int*>(sectors_buf.ptr);
    int* unique_ptr = static_cast<int*>(unique_buf.ptr);
    
    // Create mapping from sector ID to index
    std::unordered_map<int, int> sector_to_index;
    for (int i = 0; i < n_sectors; ++i) {
        sector_to_index[unique_ptr[i]] = i;
    }
    
    // Create output matrix (n_firms x n_sectors)
    auto result = py::array_t<double>(n_firms * n_sectors);
    result.resize({n_firms, n_sectors});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Initialize to zero
    std::fill(result_ptr, result_ptr + n_firms * n_sectors, 0.0);
    
    // Fill indicator matrix
    for (int i = 0; i < n_firms; ++i) {
        auto it = sector_to_index.find(sectors_ptr[i]);
        if (it != sector_to_index.end()) {
            int sector_idx = it->second;
            result_ptr[i * n_sectors + sector_idx] = 1.0;
        }
    }
    
    return result;
}

bool check_sparsity_cpp(py::array_t<double> matrix, double threshold) {
    auto buf = matrix.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input matrix must be 2-dimensional");
    }
    
    int total_elements = buf.shape[0] * buf.shape[1];
    if (total_elements == 0) return false;
    
    double* ptr = static_cast<double*>(buf.ptr);
    
    int non_zero_count = 0;
    for (int i = 0; i < total_elements; ++i) {
        if (std::abs(ptr[i]) > 1e-12) {  // Consider effectively zero
            non_zero_count++;
        }
    }
    
    double sparsity = 1.0 - static_cast<double>(non_zero_count) / total_elements;
    return sparsity > threshold;
}

py::array_t<double> sparse_matrix_multiply_cpp(py::array_t<double> A, py::array_t<double> B, double sparsity_threshold) {
    // Check if matrices are sparse enough to benefit from sparse operations
    bool A_sparse = check_sparsity_cpp(A, sparsity_threshold);
    bool B_sparse = check_sparsity_cpp(B, sparsity_threshold);
    
    if (!A_sparse && !B_sparse) {
        // Use regular dense multiplication
        return fast_matrix_multiply_cpp(A, B, true);
    }
    
    auto buf_A = A.request();
    auto buf_B = B.request();
    
    if (buf_A.ndim != 2 || buf_B.ndim != 2) {
        throw std::runtime_error("Input matrices must be 2-dimensional");
    }
    
    if (buf_A.shape[1] != buf_B.shape[0]) {
        throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
    }
    
    int m = buf_A.shape[0];
    int n = buf_B.shape[1];
    int k = buf_A.shape[1];
    
    double* ptr_A = static_cast<double*>(buf_A.ptr);
    double* ptr_B = static_cast<double*>(buf_B.ptr);
    
    // Create output matrix
    auto result = py::array_t<double>(m * n);
    result.resize({m, n});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Initialize result matrix to zero
    std::fill(result_ptr, result_ptr + m * n, 0.0);
    
    // Sparse multiplication: only compute for non-zero elements
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            double a_il = ptr_A[i * k + l];
            if (std::abs(a_il) > 1e-12) {  // Only process non-zero elements
                for (int j = 0; j < n; ++j) {
                    double b_lj = ptr_B[l * n + j];
                    if (std::abs(b_lj) > 1e-12) {
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        result_ptr[i * n + j] += a_il * b_lj;
                    }
                }
            }
        }
    }
    
    return result;
}

py::array_t<double> batch_minimum_maximum_cpp(py::array_t<double> matrix_i, py::array_t<double> matrix_j, int batch_size) {
    auto buf_i = matrix_i.request();
    auto buf_j = matrix_j.request();
    
    if (buf_i.ndim != 3 || buf_j.ndim != 3) {
        throw std::runtime_error("Input matrices must be 3-dimensional");
    }
    
    if (buf_i.shape[0] != buf_j.shape[0] || buf_i.shape[1] != buf_j.shape[1] || buf_i.shape[2] != buf_j.shape[2]) {
        throw std::runtime_error("Input matrices must have the same shape");
    }
    
    int n_sectors = buf_i.shape[0];
    int n_firms_i = buf_i.shape[1];
    int n_firms_j = buf_i.shape[2];
    
    double* ptr_i = static_cast<double*>(buf_i.ptr);
    double* ptr_j = static_cast<double*>(buf_j.ptr);
    
    // Create output matrices for minimums and maximums
    auto minimums = py::array_t<double>(n_sectors * n_firms_i * n_firms_j);
    minimums.resize({n_sectors, n_firms_i, n_firms_j});
    auto min_buf = minimums.request();
    double* min_ptr = static_cast<double*>(min_buf.ptr);
    
    auto maximums = py::array_t<double>(n_sectors * n_firms_i * n_firms_j);
    maximums.resize({n_sectors, n_firms_i, n_firms_j});
    auto max_buf = maximums.request();
    double* max_ptr = static_cast<double*>(max_buf.ptr);
    
    int total_elements = n_sectors * n_firms_i * n_firms_j;
    
    // Process in batches for memory efficiency
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, batch_size)
    #endif
    for (int idx = 0; idx < total_elements; ++idx) {
        double val_i = ptr_i[idx];
        double val_j = ptr_j[idx];
        
        min_ptr[idx] = std::min(val_i, val_j);
        max_ptr[idx] = std::max(val_i, val_j);
    }
    
    return minimums;  // Return minimums for now; in practice, you'd want to return both
}