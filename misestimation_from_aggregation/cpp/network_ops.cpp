#include "network_ops.hpp"
#include "matrix_ops.hpp"
#include <algorithm>
#include <unordered_map>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

std::tuple<py::array_t<int>, std::vector<int>, std::vector<int>> get_sector_mapping_cpp(py::array_t<int> sectors) {
    auto buf = sectors.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Sectors array must be 1-dimensional");
    }
    
    int n_firms = buf.shape[0];
    int* sectors_ptr = static_cast<int*>(buf.ptr);
    
    // Find unique sectors and create mapping
    std::set<int> unique_set;
    for (int i = 0; i < n_firms; ++i) {
        unique_set.insert(sectors_ptr[i]);
    }
    
    std::vector<int> unique_sectors(unique_set.begin(), unique_set.end());
    std::sort(unique_sectors.begin(), unique_sectors.end());
    
    int n_sectors = unique_sectors.size();
    
    // Create sector to index mapping
    std::unordered_map<int, int> sector_to_index;
    for (int i = 0; i < n_sectors; ++i) {
        sector_to_index[unique_sectors[i]] = i;
    }
    
    // Create consecutive sectors array
    auto consecutive_sectors = py::array_t<int>(n_firms);
    auto consec_buf = consecutive_sectors.request();
    int* consec_ptr = static_cast<int*>(consec_buf.ptr);
    
    for (int i = 0; i < n_firms; ++i) {
        consec_ptr[i] = sector_to_index[sectors_ptr[i]];
    }
    
    // Create index to sector mapping (reverse mapping)
    std::vector<int> index_to_sector = unique_sectors;
    
    return std::make_tuple(consecutive_sectors, unique_sectors, index_to_sector);
}

py::array_t<double> aggregate_to_sectors_cpp(py::array_t<double> network, py::array_t<int> sectors, bool use_parallel) {
    auto network_buf = network.request();
    auto sectors_buf = sectors.request();
    
    if (network_buf.ndim != 2) {
        throw std::runtime_error("Network must be 2-dimensional");
    }
    
    if (sectors_buf.ndim != 1) {
        throw std::runtime_error("Sectors must be 1-dimensional");
    }
    
    int n_firms = network_buf.shape[0];
    if (n_firms != network_buf.shape[1]) {
        throw std::runtime_error("Network must be square matrix");
    }
    
    if (n_firms != sectors_buf.shape[0]) {
        throw std::runtime_error("Number of firms in network and sectors must match");
    }
    
    // Get sector mapping
    auto [consecutive_sectors, unique_sectors_vec, index_to_sector] = get_sector_mapping_cpp(sectors);
    int n_sectors = unique_sectors_vec.size();
    
    // Create unique_sectors array for indicator matrix
    auto unique_sectors_array = py::array_t<int>(n_sectors);
    auto unique_buf = unique_sectors_array.request();
    int* unique_ptr = static_cast<int*>(unique_buf.ptr);
    for (int i = 0; i < n_sectors; ++i) {
        unique_ptr[i] = i;  // Use consecutive indices
    }
    
    // Create indicator matrix P (n_firms x n_sectors)
    auto P = optimized_indicator_matrix_cpp(consecutive_sectors, unique_sectors_array);
    
    // Transpose P to get P^T (n_sectors x n_firms)
    auto P_buf = P.request();
    auto P_T = py::array_t<double>(n_sectors * n_firms);
    P_T.resize({n_sectors, n_firms});
    auto PT_buf = P_T.request();
    
    double* P_ptr = static_cast<double*>(P_buf.ptr);
    double* PT_ptr = static_cast<double*>(PT_buf.ptr);
    
    // Transpose P: P(i,j) -> P_T(j,i)
    for (int i = 0; i < n_firms; ++i) {
        for (int j = 0; j < n_sectors; ++j) {
            PT_ptr[j * n_firms + i] = P_ptr[i * n_sectors + j];
        }
    }
    
    // Compute P^T * W
    auto temp = fast_matrix_multiply_cpp(P_T, network, use_parallel);
    
    // Compute (P^T * W) * P = P^T * W * P
    auto result = fast_matrix_multiply_cpp(temp, P, use_parallel);
    
    return result;
}

std::pair<py::array_t<double>, py::array_t<double>> aggregate_suppliers_cpp(
    py::array_t<double> network, py::array_t<int> sectors, py::array_t<int> unique_sectors) {
    
    auto network_buf = network.request();
    int n_firms = network_buf.shape[0];
    
    // Create indicator matrix P (n_firms x n_sectors)
    auto P = optimized_indicator_matrix_cpp(sectors, unique_sectors);
    auto P_buf = P.request();
    int n_sectors = P_buf.shape[1];
    
    // Transpose P to get P^T (n_sectors x n_firms)
    auto P_T = py::array_t<double>(n_sectors * n_firms);
    P_T.resize({n_sectors, n_firms});
    auto PT_buf = P_T.request();
    
    double* P_ptr = static_cast<double*>(P_buf.ptr);
    double* PT_ptr = static_cast<double*>(PT_buf.ptr);
    
    // Transpose P
    for (int i = 0; i < n_firms; ++i) {
        for (int j = 0; j < n_sectors; ++j) {
            PT_ptr[j * n_firms + i] = P_ptr[i * n_sectors + j];
        }
    }
    
    // Create binary network for counts
    auto binary_network = py::array_t<double>(n_firms * n_firms);
    binary_network.resize({n_firms, n_firms});
    auto binary_buf = binary_network.request();
    auto net_buf = network.request();
    
    double* binary_ptr = static_cast<double*>(binary_buf.ptr);
    double* net_ptr = static_cast<double*>(net_buf.ptr);
    
    // Convert to binary (connections)
    for (int i = 0; i < n_firms * n_firms; ++i) {
        binary_ptr[i] = (net_ptr[i] > 0.0) ? 1.0 : 0.0;
    }
    
    // Aggregate: P^T * binary_network for counts, P^T * network for volumes
    auto counts = fast_matrix_multiply_cpp(P_T, binary_network, true);
    auto volumes = fast_matrix_multiply_cpp(P_T, network, true);
    
    return std::make_pair(counts, volumes);
}

std::pair<py::array_t<double>, py::array_t<double>> aggregate_buyers_cpp(
    py::array_t<double> network, py::array_t<int> sectors, py::array_t<int> unique_sectors) {
    
    auto network_buf = network.request();
    int n_firms = network_buf.shape[0];
    
    // Create indicator matrix P (n_firms x n_sectors)
    auto P = optimized_indicator_matrix_cpp(sectors, unique_sectors);
    auto P_buf = P.request();
    int n_sectors = P_buf.shape[1];
    
    // Create binary network for counts
    auto binary_network = py::array_t<double>(n_firms * n_firms);
    binary_network.resize({n_firms, n_firms});
    auto binary_buf = binary_network.request();
    auto net_buf = network.request();
    
    double* binary_ptr = static_cast<double*>(binary_buf.ptr);
    double* net_ptr = static_cast<double*>(net_buf.ptr);
    
    // Convert to binary (connections)
    for (int i = 0; i < n_firms * n_firms; ++i) {
        binary_ptr[i] = (net_ptr[i] > 0.0) ? 1.0 : 0.0;
    }
    
    // Aggregate: binary_network * P for counts, network * P for volumes
    auto counts_temp = fast_matrix_multiply_cpp(binary_network, P, true);
    auto volumes_temp = fast_matrix_multiply_cpp(network, P, true);
    
    // Transpose results to get (n_sectors x n_firms)
    auto counts = py::array_t<double>(n_sectors * n_firms);
    counts.resize({n_sectors, n_firms});
    auto counts_buf = counts.request();
    auto counts_temp_buf = counts_temp.request();
    
    auto volumes = py::array_t<double>(n_sectors * n_firms);
    volumes.resize({n_sectors, n_firms});
    auto volumes_buf = volumes.request();
    auto volumes_temp_buf = volumes_temp.request();
    
    double* counts_ptr = static_cast<double*>(counts_buf.ptr);
    double* counts_temp_ptr = static_cast<double*>(counts_temp_buf.ptr);
    double* volumes_ptr = static_cast<double*>(volumes_buf.ptr);
    double* volumes_temp_ptr = static_cast<double*>(volumes_temp_buf.ptr);
    
    // Transpose: (n_firms x n_sectors) -> (n_sectors x n_firms)
    for (int i = 0; i < n_firms; ++i) {
        for (int j = 0; j < n_sectors; ++j) {
            counts_ptr[j * n_firms + i] = counts_temp_ptr[i * n_sectors + j];
            volumes_ptr[j * n_firms + i] = volumes_temp_ptr[i * n_sectors + j];
        }
    }
    
    return std::make_pair(counts, volumes);
}