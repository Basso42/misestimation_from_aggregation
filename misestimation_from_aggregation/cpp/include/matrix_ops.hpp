#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <numpy/arrayobject.h>
#include <vector>

namespace py = pybind11;

/**
 * C++ optimized matrix operations for misestimation_from_aggregation
 */

// Fast matrix multiplication with automatic sparsity detection
py::array_t<double> fast_matrix_multiply_cpp(
    py::array_t<double> A,
    py::array_t<double> B,
    bool use_parallel = true
);

// Optimized indicator matrix creation
py::array_t<double> optimized_indicator_matrix_cpp(
    py::array_t<int> sectors,
    py::array_t<int> unique_sectors
);

// Sparse matrix multiplication (for very sparse matrices)
py::array_t<double> sparse_matrix_multiply_cpp(
    py::array_t<double> A,
    py::array_t<double> B,
    double sparsity_threshold = 0.1
);

// Check if matrix is sparse enough to benefit from sparse operations
bool check_sparsity_cpp(
    py::array_t<double> matrix,
    double threshold = 0.1
);

// Memory-efficient batch processing for large matrices
py::array_t<double> batch_minimum_maximum_cpp(
    py::array_t<double> matrix_i,
    py::array_t<double> matrix_j,
    int batch_size = 1000
);