#include "similarity_ops.hpp"
#include <cstring>
#include <algorithm>
#include <thread>
#include <future>

#ifdef _OPENMP
#include <omp.h>
#endif

double safe_divide(double numerator, double denominator, double fill_value) {
    if (denominator == 0.0 || std::isnan(denominator)) {
        return fill_value;
    }
    return numerator / denominator;
}

py::array_t<double> pairwise_weighted_jaccard_cpp(py::array_t<double> matrix, bool use_parallel) {
    auto buf = matrix.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input matrix must be 2-dimensional");
    }
    
    int n_sectors = buf.shape[0];
    int n_firms = buf.shape[1];
    
    // Create output matrix
    auto result = py::array_t<double>(n_firms * n_firms);
    result.resize({n_firms, n_firms});
    auto result_buf = result.request();
    
    double* input_ptr = static_cast<double*>(buf.ptr);
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Initialize result matrix to zero
    std::fill(result_ptr, result_ptr + n_firms * n_firms, 0.0);
    
    // Parallel computation of pairwise weighted Jaccard indices
    #ifdef _OPENMP
    if (use_parallel) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < n_firms; ++i) {
            for (int j = 0; j < n_firms; ++j) {
                double intersection = 0.0;
                double union_sum = 0.0;
                
                for (int k = 0; k < n_sectors; ++k) {
                    double val_i = input_ptr[k * n_firms + i];
                    double val_j = input_ptr[k * n_firms + j];
                    
                    intersection += std::min(val_i, val_j);
                    union_sum += std::max(val_i, val_j);
                }
                
                result_ptr[i * n_firms + j] = safe_divide(intersection, union_sum, 0.0);
            }
        }
    } else
    #endif
    {
        // Sequential version
        for (int i = 0; i < n_firms; ++i) {
            for (int j = 0; j < n_firms; ++j) {
                double intersection = 0.0;
                double union_sum = 0.0;
                
                for (int k = 0; k < n_sectors; ++k) {
                    double val_i = input_ptr[k * n_firms + i];
                    double val_j = input_ptr[k * n_firms + j];
                    
                    intersection += std::min(val_i, val_j);
                    union_sum += std::max(val_i, val_j);
                }
                
                result_ptr[i * n_firms + j] = safe_divide(intersection, union_sum, 0.0);
            }
        }
    }
    
    return result;
}

py::array_t<double> pairwise_overlap_relative_cpp(py::array_t<double> matrix, bool use_parallel) {
    auto buf = matrix.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input matrix must be 2-dimensional");
    }
    
    int n_sectors = buf.shape[0];
    int n_firms = buf.shape[1];
    
    // First normalize the matrix columns
    auto normalized = py::array_t<double>(n_sectors * n_firms);
    normalized.resize({n_sectors, n_firms});
    auto norm_buf = normalized.request();
    
    double* input_ptr = static_cast<double*>(buf.ptr);
    double* norm_ptr = static_cast<double*>(norm_buf.ptr);
    
    // Normalize columns by their sums
    for (int j = 0; j < n_firms; ++j) {
        double column_sum = 0.0;
        for (int i = 0; i < n_sectors; ++i) {
            column_sum += input_ptr[i * n_firms + j];
        }
        
        for (int i = 0; i < n_sectors; ++i) {
            norm_ptr[i * n_firms + j] = safe_divide(input_ptr[i * n_firms + j], column_sum, 0.0);
        }
    }
    
    // Create output matrix
    auto result = py::array_t<double>(n_firms * n_firms);
    result.resize({n_firms, n_firms});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Initialize result matrix to zero
    std::fill(result_ptr, result_ptr + n_firms * n_firms, 0.0);
    
    // Parallel computation of pairwise overlaps
    #ifdef _OPENMP
    if (use_parallel) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < n_firms; ++i) {
            for (int j = 0; j < n_firms; ++j) {
                double overlap = 0.0;
                
                for (int k = 0; k < n_sectors; ++k) {
                    double val_i = norm_ptr[k * n_firms + i];
                    double val_j = norm_ptr[k * n_firms + j];
                    
                    overlap += std::min(val_i, val_j);
                }
                
                result_ptr[i * n_firms + j] = overlap;
            }
        }
    } else
    #endif
    {
        // Sequential version
        for (int i = 0; i < n_firms; ++i) {
            for (int j = 0; j < n_firms; ++j) {
                double overlap = 0.0;
                
                for (int k = 0; k < n_sectors; ++k) {
                    double val_i = norm_ptr[k * n_firms + i];
                    double val_j = norm_ptr[k * n_firms + j];
                    
                    overlap += std::min(val_i, val_j);
                }
                
                result_ptr[i * n_firms + j] = overlap;
            }
        }
    }
    
    return result;
}

py::array_t<double> pairwise_cosine_similarity_cpp(py::array_t<double> matrix, bool use_parallel) {
    auto buf = matrix.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input matrix must be 2-dimensional");
    }
    
    int n_sectors = buf.shape[0];
    int n_firms = buf.shape[1];
    
    double* input_ptr = static_cast<double*>(buf.ptr);
    
    // Pre-compute norms for all firms
    std::vector<double> norms(n_firms);
    for (int j = 0; j < n_firms; ++j) {
        double norm_sq = 0.0;
        for (int i = 0; i < n_sectors; ++i) {
            double val = input_ptr[i * n_firms + j];
            norm_sq += val * val;
        }
        norms[j] = std::sqrt(norm_sq);
    }
    
    // Create output matrix
    auto result = py::array_t<double>(n_firms * n_firms);
    result.resize({n_firms, n_firms});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Initialize result matrix to zero
    std::fill(result_ptr, result_ptr + n_firms * n_firms, 0.0);
    
    // Parallel computation of pairwise cosine similarities
    #ifdef _OPENMP
    if (use_parallel) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < n_firms; ++i) {
            for (int j = 0; j < n_firms; ++j) {
                double dot_product = 0.0;
                
                for (int k = 0; k < n_sectors; ++k) {
                    double val_i = input_ptr[k * n_firms + i];
                    double val_j = input_ptr[k * n_firms + j];
                    
                    dot_product += val_i * val_j;
                }
                
                double norm_product = norms[i] * norms[j];
                result_ptr[i * n_firms + j] = safe_divide(dot_product, norm_product, 0.0);
            }
        }
    } else
    #endif
    {
        // Sequential version
        for (int i = 0; i < n_firms; ++i) {
            for (int j = 0; j < n_firms; ++j) {
                double dot_product = 0.0;
                
                for (int k = 0; k < n_sectors; ++k) {
                    double val_i = input_ptr[k * n_firms + i];
                    double val_j = input_ptr[k * n_firms + j];
                    
                    dot_product += val_i * val_j;
                }
                
                double norm_product = norms[i] * norms[j];
                result_ptr[i * n_firms + j] = safe_divide(dot_product, norm_product, 0.0);
            }
        }
    }
    
    return result;
}

py::array_t<double> temporal_weighted_jaccard_cpp(py::array_t<double> matrix_t, py::array_t<double> matrix_tm1, bool use_parallel) {
    auto buf_t = matrix_t.request();
    auto buf_tm1 = matrix_tm1.request();
    
    if (buf_t.ndim != 2 || buf_tm1.ndim != 2) {
        throw std::runtime_error("Input matrices must be 2-dimensional");
    }
    
    if (buf_t.shape[0] != buf_tm1.shape[0] || buf_t.shape[1] != buf_tm1.shape[1]) {
        throw std::runtime_error("Input matrices must have the same shape");
    }
    
    int n_sectors = buf_t.shape[0];
    int n_firms = buf_t.shape[1];
    
    double* ptr_t = static_cast<double*>(buf_t.ptr);
    double* ptr_tm1 = static_cast<double*>(buf_tm1.ptr);
    
    // Create output vector
    auto result = py::array_t<double>(n_firms);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Parallel computation of temporal weighted Jaccard
    #ifdef _OPENMP
    if (use_parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < n_firms; ++j) {
            double intersection = 0.0;
            double union_sum = 0.0;
            
            for (int i = 0; i < n_sectors; ++i) {
                double val_t = ptr_t[i * n_firms + j];
                double val_tm1 = ptr_tm1[i * n_firms + j];
                
                intersection += std::min(val_t, val_tm1);
                union_sum += std::max(val_t, val_tm1);
            }
            
            result_ptr[j] = safe_divide(intersection, union_sum, 0.0);
        }
    } else
    #endif
    {
        // Sequential version
        for (int j = 0; j < n_firms; ++j) {
            double intersection = 0.0;
            double union_sum = 0.0;
            
            for (int i = 0; i < n_sectors; ++i) {
                double val_t = ptr_t[i * n_firms + j];
                double val_tm1 = ptr_tm1[i * n_firms + j];
                
                intersection += std::min(val_t, val_tm1);
                union_sum += std::max(val_t, val_tm1);
            }
            
            result_ptr[j] = safe_divide(intersection, union_sum, 0.0);
        }
    }
    
    return result;
}