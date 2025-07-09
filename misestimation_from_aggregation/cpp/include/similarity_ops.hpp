#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

/**
 * C++ optimized similarity calculations for misestimation_from_aggregation
 */

// Pairwise weighted Jaccard similarity calculation
py::array_t<double> pairwise_weighted_jaccard_cpp(
    py::array_t<double> matrix,
    bool use_parallel = true
);

// Pairwise overlap relative calculation  
py::array_t<double> pairwise_overlap_relative_cpp(
    py::array_t<double> matrix,
    bool use_parallel = true
);

// Pairwise cosine similarity calculation
py::array_t<double> pairwise_cosine_similarity_cpp(
    py::array_t<double> matrix,
    bool use_parallel = true
);

// Temporal weighted Jaccard similarity
py::array_t<double> temporal_weighted_jaccard_cpp(
    py::array_t<double> matrix_t,
    py::array_t<double> matrix_tm1,
    bool use_parallel = true
);

// Helper functions for safe division and memory-efficient operations
double safe_divide(double numerator, double denominator, double fill_value = 0.0);