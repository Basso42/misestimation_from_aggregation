#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <unordered_map>

namespace py = pybind11;

/**
 * C++ optimized network aggregation operations for misestimation_from_aggregation
 */

// Network aggregation to sectors
py::array_t<double> aggregate_to_sectors_cpp(
    py::array_t<double> network,
    py::array_t<int> sectors,
    bool use_parallel = true
);

// Aggregate suppliers (sectors x firms matrix)
std::pair<py::array_t<double>, py::array_t<double>> aggregate_suppliers_cpp(
    py::array_t<double> network,
    py::array_t<int> sectors,
    py::array_t<int> unique_sectors
);

// Aggregate buyers (sectors x firms matrix)
std::pair<py::array_t<double>, py::array_t<double>> aggregate_buyers_cpp(
    py::array_t<double> network, 
    py::array_t<int> sectors,
    py::array_t<int> unique_sectors
);

// Helper function to get sector mapping efficiently
std::tuple<py::array_t<int>, std::vector<int>, std::vector<int>> get_sector_mapping_cpp(
    py::array_t<int> sectors
);