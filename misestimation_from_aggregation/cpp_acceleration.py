"""
C++ acceleration module for misestimation_from_aggregation.

This module provides C++ accelerated implementations of core computational functions.
It automatically falls back to pure Python implementations if C++ extensions are not available.
"""

import warnings

# Try to import C++ extensions
try:
    from . import _cpp_core
    HAS_CPP_EXTENSIONS = True
    _cpp_available = True
except ImportError as e:
    HAS_CPP_EXTENSIONS = False
    _cpp_available = False
    _cpp_core = None
    warnings.warn(
        f"C++ extensions not available, falling back to Python implementations. "
        f"For maximum performance, ensure C++ extensions are compiled. Error: {e}", 
        UserWarning, 
        stacklevel=2
    )

def is_cpp_available():
    """Check if C++ extensions are available."""
    return _cpp_available

def get_cpp_info():
    """Get information about C++ extensions availability."""
    if _cpp_available:
        return {
            "available": True,
            "module": _cpp_core,
            "functions": [
                "pairwise_weighted_jaccard_cpp",
                "pairwise_overlap_relative_cpp", 
                "pairwise_cosine_similarity_cpp",
                "temporal_weighted_jaccard_cpp",
                "fast_matrix_multiply_cpp",
                "optimized_indicator_matrix_cpp",
                "aggregate_to_sectors_cpp",
                "aggregate_suppliers_cpp",
                "aggregate_buyers_cpp",
            ]
        }
    else:
        return {
            "available": False,
            "module": None,
            "functions": []
        }

# Export the C++ core module if available
__all__ = ["HAS_CPP_EXTENSIONS", "is_cpp_available", "get_cpp_info"]

if HAS_CPP_EXTENSIONS:
    __all__.extend([
        "pairwise_weighted_jaccard_cpp",
        "pairwise_overlap_relative_cpp", 
        "pairwise_cosine_similarity_cpp",
        "temporal_weighted_jaccard_cpp",
        "fast_matrix_multiply_cpp",
        "optimized_indicator_matrix_cpp",
        "aggregate_to_sectors_cpp", 
        "aggregate_suppliers_cpp",
        "aggregate_buyers_cpp",
        "safe_divide",
        "check_sparsity_cpp",
    ])
    
    # Re-export C++ functions
    pairwise_weighted_jaccard_cpp = _cpp_core.pairwise_weighted_jaccard_cpp
    pairwise_overlap_relative_cpp = _cpp_core.pairwise_overlap_relative_cpp
    pairwise_cosine_similarity_cpp = _cpp_core.pairwise_cosine_similarity_cpp
    temporal_weighted_jaccard_cpp = _cpp_core.temporal_weighted_jaccard_cpp
    fast_matrix_multiply_cpp = _cpp_core.fast_matrix_multiply_cpp
    optimized_indicator_matrix_cpp = _cpp_core.optimized_indicator_matrix_cpp
    aggregate_to_sectors_cpp = _cpp_core.aggregate_to_sectors_cpp
    aggregate_suppliers_cpp = _cpp_core.aggregate_suppliers_cpp
    aggregate_buyers_cpp = _cpp_core.aggregate_buyers_cpp
    safe_divide = _cpp_core.safe_divide
    check_sparsity_cpp = _cpp_core.check_sparsity_cpp