#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "include/similarity_ops.hpp"
#include "include/matrix_ops.hpp"
#include "include/network_ops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_core, m) {
    m.doc() = "C++ optimized core functions for misestimation_from_aggregation";
    
    // Similarity operations
    m.def("pairwise_weighted_jaccard_cpp", &pairwise_weighted_jaccard_cpp,
          "Compute pairwise weighted Jaccard similarity matrix",
          py::arg("matrix"), py::arg("use_parallel") = true);
    
    m.def("pairwise_overlap_relative_cpp", &pairwise_overlap_relative_cpp,
          "Compute pairwise relative overlap matrix",
          py::arg("matrix"), py::arg("use_parallel") = true);
    
    m.def("pairwise_cosine_similarity_cpp", &pairwise_cosine_similarity_cpp,
          "Compute pairwise cosine similarity matrix",
          py::arg("matrix"), py::arg("use_parallel") = true);
    
    m.def("temporal_weighted_jaccard_cpp", &temporal_weighted_jaccard_cpp,
          "Compute temporal weighted Jaccard similarities",
          py::arg("matrix_t"), py::arg("matrix_tm1"), py::arg("use_parallel") = true);
    
    m.def("safe_divide", &safe_divide,
          "Safe division with fill value for zero denominator",
          py::arg("numerator"), py::arg("denominator"), py::arg("fill_value") = 0.0);
    
    // Matrix operations
    m.def("fast_matrix_multiply_cpp", &fast_matrix_multiply_cpp,
          "Fast matrix multiplication with automatic optimization",
          py::arg("A"), py::arg("B"), py::arg("use_parallel") = true);
    
    m.def("optimized_indicator_matrix_cpp", &optimized_indicator_matrix_cpp,
          "Create optimized indicator matrix for sector aggregation",
          py::arg("sectors"), py::arg("unique_sectors"));
    
    m.def("sparse_matrix_multiply_cpp", &sparse_matrix_multiply_cpp,
          "Sparse-optimized matrix multiplication",
          py::arg("A"), py::arg("B"), py::arg("sparsity_threshold") = 0.1);
    
    m.def("check_sparsity_cpp", &check_sparsity_cpp,
          "Check if matrix is sparse enough for sparse operations",
          py::arg("matrix"), py::arg("threshold") = 0.1);
    
    m.def("batch_minimum_maximum_cpp", &batch_minimum_maximum_cpp,
          "Memory-efficient batch processing for minimum/maximum operations",
          py::arg("matrix_i"), py::arg("matrix_j"), py::arg("batch_size") = 1000);
    
    // Network operations
    m.def("aggregate_to_sectors_cpp", &aggregate_to_sectors_cpp,
          "Aggregate firm-level network to sector-level network",
          py::arg("network"), py::arg("sectors"), py::arg("use_parallel") = true);
    
    m.def("aggregate_suppliers_cpp", &aggregate_suppliers_cpp,
          "Aggregate suppliers to sectors (returns counts and volumes)",
          py::arg("network"), py::arg("sectors"), py::arg("unique_sectors"));
    
    m.def("aggregate_buyers_cpp", &aggregate_buyers_cpp,
          "Aggregate buyers to sectors (returns counts and volumes)",
          py::arg("network"), py::arg("sectors"), py::arg("unique_sectors"));
    
    m.def("get_sector_mapping_cpp", &get_sector_mapping_cpp,
          "Get efficient sector mapping for aggregation operations",
          py::arg("sectors"));
}