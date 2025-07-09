"""
Utility functions for network validation and data processing.
"""

import numpy as np
from typing import Union, List, Tuple
import warnings


def validate_network(network: np.ndarray, allow_negative: bool = False) -> np.ndarray:
    """
    Validate and clean network adjacency matrix.
    
    Parameters
    ----------
    network : np.ndarray
        Network adjacency matrix, shape (n_firms, n_firms)
    allow_negative : bool, default False
        Whether to allow negative weights
        
    Returns
    -------
    np.ndarray
        Validated network matrix
        
    Raises
    ------
    ValueError
        If network is not a valid adjacency matrix
    """
    network = np.asarray(network)
    
    if network.ndim != 2:
        raise ValueError("Network must be a 2D matrix")
    
    if network.shape[0] != network.shape[1]:
        raise ValueError("Network must be square (n_firms x n_firms)")
    
    if not allow_negative and np.any(network < 0):
        raise ValueError("Network contains negative weights. Set allow_negative=True if intended.")
    
    if not np.isfinite(network).all():
        raise ValueError("Network contains non-finite values (NaN or inf)")
    
    return network


def validate_sectors(sectors: Union[List, np.ndarray], n_firms: int) -> np.ndarray:
    """
    Validate sector affiliation vector.
    
    Parameters
    ----------
    sectors : array-like
        Sector affiliation for each firm
    n_firms : int
        Expected number of firms
        
    Returns
    -------
    np.ndarray
        Validated sector vector
        
    Raises
    ------
    ValueError
        If sectors vector is invalid
    """
    sectors = np.asarray(sectors)
    
    if sectors.ndim != 1:
        raise ValueError("Sectors must be a 1D array")
    
    if len(sectors) != n_firms:
        raise ValueError(f"Sectors length ({len(sectors)}) must match number of firms ({n_firms})")
    
    if not np.isfinite(sectors).all():
        raise ValueError("Sectors contains non-finite values")
    
    return sectors


def get_sector_mapping(sectors: np.ndarray) -> Tuple[np.ndarray, dict, dict]:
    """
    Create mapping between original sector labels and consecutive indices.
    
    Parameters
    ----------
    sectors : np.ndarray
        Sector affiliation vector
        
    Returns
    -------
    tuple
        (consecutive_sectors, sector_to_index, index_to_sector)
    """
    unique_sectors = np.unique(sectors)
    sector_to_index = {sector: i for i, sector in enumerate(unique_sectors)}
    index_to_sector = {i: sector for i, sector in enumerate(unique_sectors)}
    
    consecutive_sectors = np.array([sector_to_index[s] for s in sectors])
    
    return consecutive_sectors, sector_to_index, index_to_sector


def check_sparsity(matrix: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Check if matrix is sparse (has many zeros).
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    threshold : float, default 0.1
        Sparsity threshold (fraction of non-zero elements)
        
    Returns
    -------
    bool
        True if matrix is sparse (< threshold non-zero elements)
    """
    if matrix.size == 0:
        return True
    
    # Handle sparse matrices
    try:
        from scipy import sparse
        if sparse.issparse(matrix):
            non_zero_fraction = matrix.nnz / matrix.size
            return non_zero_fraction < threshold
    except ImportError:
        pass
    
    # Handle dense arrays
    if isinstance(matrix, np.ndarray):
        non_zero_fraction = np.count_nonzero(matrix) / matrix.size
        return non_zero_fraction < threshold
    
    # Fallback for other matrix types
    try:
        non_zero_fraction = np.count_nonzero(matrix) / matrix.size
        return non_zero_fraction < threshold
    except:
        return False


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                fill_value: float = 0.0) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array
    denominator : np.ndarray  
        Denominator array
    fill_value : float, default 0.0
        Value to use when denominator is zero
        
    Returns
    -------
    np.ndarray
        Result of safe division
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.divide(numerator, denominator, 
                          out=np.full_like(numerator, fill_value, dtype=float),
                          where=denominator != 0)
    return result


def create_sparse_matrix(rows: np.ndarray, cols: np.ndarray, 
                        data: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Create dense matrix from sparse representation using optimized indexing.
    
    Parameters
    ----------
    rows : np.ndarray
        Row indices
    cols : np.ndarray
        Column indices  
    data : np.ndarray
        Data values
    shape : tuple
        Matrix shape (n_rows, n_cols)
        
    Returns
    -------
    np.ndarray
        Dense matrix
    """
    # Use optimized sparse matrix creation if available
    try:
        from scipy import sparse
        sparse_matrix = sparse.coo_matrix((data, (rows, cols)), shape=shape)
        return sparse_matrix.toarray()
    except ImportError:
        # Fallback to numpy
        matrix = np.zeros(shape, dtype=data.dtype)
        matrix[rows, cols] = data
        return matrix


def vectorized_sector_aggregation(values: np.ndarray, sectors: np.ndarray, 
                                 operation: str = 'sum') -> np.ndarray:
    """
    Efficiently aggregate values by sector using vectorized operations.
    
    Parameters
    ----------
    values : np.ndarray
        Values to aggregate (one per firm)
    sectors : np.ndarray
        Sector assignments for each firm
    operation : str, default 'sum'
        Aggregation operation ('sum', 'mean', 'count')
        
    Returns
    -------
    np.ndarray
        Aggregated values by sector
    """
    unique_sectors = np.unique(sectors)
    n_sectors = len(unique_sectors)
    
    if operation == 'sum':
        # Use bincount for efficient summation
        sector_indices = np.searchsorted(unique_sectors, sectors)
        return np.bincount(sector_indices, weights=values, minlength=n_sectors)
    
    elif operation == 'mean':
        # Calculate sum and count, then divide
        sector_indices = np.searchsorted(unique_sectors, sectors)
        sector_sums = np.bincount(sector_indices, weights=values, minlength=n_sectors)
        sector_counts = np.bincount(sector_indices, minlength=n_sectors)
        return safe_divide(sector_sums, sector_counts, fill_value=0.0)
    
    elif operation == 'count':
        sector_indices = np.searchsorted(unique_sectors, sectors)
        return np.bincount(sector_indices, minlength=n_sectors)
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


def fast_matrix_multiply(A: np.ndarray, B: np.ndarray, 
                        use_blas: bool = True) -> np.ndarray:
    """
    Fast matrix multiplication with automatic optimization selection.
    
    Parameters
    ----------
    A : np.ndarray
        First matrix
    B : np.ndarray
        Second matrix
    use_blas : bool, default True
        Whether to use BLAS optimizations when available
        
    Returns
    -------
    np.ndarray
        Matrix product A @ B
    """
    try:
        from scipy import sparse
        
        # Handle sparse matrices
        if sparse.issparse(A) or sparse.issparse(B):
            result = A @ B
            return result.toarray() if hasattr(result, 'toarray') else result
        
        # Check if sparse matrices would be beneficial for dense inputs
        if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
            if check_sparsity(A) or check_sparsity(B):
                A_sparse = sparse.csr_matrix(A) if check_sparsity(A) else A
                B_sparse = sparse.csr_matrix(B) if check_sparsity(B) else B
                
                result = A_sparse @ B_sparse
                return result.toarray() if hasattr(result, 'toarray') else result
                
    except ImportError:
        pass
    
    # Use standard numpy matrix multiplication (which uses BLAS when available)
    return A @ B