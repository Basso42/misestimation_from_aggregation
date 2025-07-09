"""
Performance optimization utilities for the misestimation_from_aggregation package.
"""

import numpy as np
from functools import lru_cache
from typing import Tuple, Optional, Dict, Any

try:
    from scipy import sparse
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False


class PerformanceCache:
    """
    Caching mechanism for expensive computations.
    """
    
    def __init__(self, max_cache_size: int = 128):
        self.max_cache_size = max_cache_size
        self._cache = {}
        
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(f"array_{arg.shape}_{hash(arg.tobytes())}")
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, np.ndarray):
                key_parts.append(f"{k}_array_{v.shape}_{hash(v.tobytes())}")
            else:
                key_parts.append(f"{k}_{v}")
        
        return "_".join(key_parts)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        return self._cache.get(key)
        
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value


# Global cache instance
_performance_cache = PerformanceCache()


def optimized_indicator_matrix(sectors: np.ndarray, unique_sectors: np.ndarray) -> np.ndarray:
    """
    Create indicator matrix efficiently using vectorized operations.
    
    Parameters
    ----------
    sectors : np.ndarray
        Sector affiliation for each firm
    unique_sectors : np.ndarray
        Unique sector identifiers
        
    Returns
    -------
    np.ndarray
        Indicator matrix of shape (n_firms, n_sectors)
    """
    cache_key = _performance_cache.get_cache_key(sectors, unique_sectors)
    cached_result = _performance_cache.get(cache_key)
    
    if cached_result is not None:
        return cached_result
    
    n_firms = len(sectors)
    n_sectors = len(unique_sectors)
    
    # Vectorized indicator matrix creation
    firm_indices = np.arange(n_firms)
    sector_indices = np.searchsorted(unique_sectors, sectors)
    
    if HAS_SCIPY_SPARSE and n_firms > 1000 and n_sectors > 100:
        # Use sparse matrix for large problems
        indicator_matrix = sparse.csr_matrix(
            (np.ones(n_firms), (firm_indices, sector_indices)), 
            shape=(n_firms, n_sectors)
        ).toarray()
    else:
        indicator_matrix = np.zeros((n_firms, n_sectors))
        indicator_matrix[firm_indices, sector_indices] = 1
    
    _performance_cache.set(cache_key, indicator_matrix)
    return indicator_matrix


def batch_minimum_maximum(matrix_i: np.ndarray, matrix_j: np.ndarray, 
                         batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute element-wise minimum and maximum in batches to manage memory.
    
    Parameters
    ----------
    matrix_i : np.ndarray
        First matrix for comparison
    matrix_j : np.ndarray  
        Second matrix for comparison
    batch_size : int
        Size of batches for processing
        
    Returns
    -------
    tuple
        (minimums, maximums) arrays
    """
    n_pairs = matrix_i.shape[-1]
    
    if n_pairs <= batch_size:
        return np.minimum(matrix_i, matrix_j), np.maximum(matrix_i, matrix_j)
    
    # Process in batches
    minimums = []
    maximums = []
    
    for i in range(0, n_pairs, batch_size):
        end_idx = min(i + batch_size, n_pairs)
        batch_i = matrix_i[..., i:end_idx]
        batch_j = matrix_j[..., i:end_idx]
        
        minimums.append(np.minimum(batch_i, batch_j))
        maximums.append(np.maximum(batch_i, batch_j))
    
    return np.concatenate(minimums, axis=-1), np.concatenate(maximums, axis=-1)


@lru_cache(maxsize=32)
def cached_unique_sectors(sectors_tuple: tuple) -> np.ndarray:
    """
    Cached version of np.unique for sectors.
    
    Parameters
    ----------
    sectors_tuple : tuple
        Sectors as tuple for hashing
        
    Returns
    -------
    np.ndarray
        Unique sectors
    """
    return np.unique(np.array(sectors_tuple))


def estimate_memory_usage(matrix_shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> float:
    """
    Estimate memory usage in MB for a matrix.
    
    Parameters
    ----------
    matrix_shape : tuple
        Shape of the matrix
    dtype : np.dtype
        Data type of the matrix
        
    Returns
    -------
    float
        Estimated memory usage in MB
    """
    elements = np.prod(matrix_shape)
    bytes_per_element = np.dtype(dtype).itemsize
    return (elements * bytes_per_element) / (1024 * 1024)


def should_use_sparse(matrix: np.ndarray, sparsity_threshold: float = 0.1) -> bool:
    """
    Determine if sparse matrix representation would be beneficial.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    sparsity_threshold : float
        Threshold for sparsity (fraction of non-zero elements)
        
    Returns
    -------
    bool
        True if sparse representation recommended
    """
    if not HAS_SCIPY_SPARSE:
        return False
    
    if matrix.size == 0:
        return False
    
    non_zero_fraction = np.count_nonzero(matrix) / matrix.size
    memory_dense = estimate_memory_usage(matrix.shape, matrix.dtype)
    
    # Use sparse if less than threshold non-zero and matrix is reasonably large
    return non_zero_fraction < sparsity_threshold and memory_dense > 10.0  # 10MB threshold


def optimize_matrix_operations(func):
    """
    Decorator to optimize matrix operations automatically.
    """
    def wrapper(*args, **kwargs):
        # Check if we should use optimized paths
        for arg in args:
            if isinstance(arg, np.ndarray) and should_use_sparse(arg):
                kwargs['use_sparse'] = True
                break
        return func(*args, **kwargs)
    
    return wrapper