"""
Similarity measures for comparing input/output vectors between firms.
"""

import numpy as np
from typing import Union, List, Dict, Optional
from multiprocessing import Pool
import warnings
from .utils import safe_divide
from .performance import batch_minimum_maximum, estimate_memory_usage, optimize_matrix_operations

try:
    from scipy import sparse
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False


class SimilarityCalculator:
    """
    Class for calculating various similarity measures between firm input/output vectors.
    
    Supports multiple similarity measures including Jaccard index, cosine similarity,
    and overlap coefficients for both pairwise and temporal comparisons.
    """
    
    def __init__(self, n_cores: int = 1):
        """
        Initialize similarity calculator.
        
        Parameters
        ----------
        n_cores : int, default 1
            Number of CPU cores to use for parallel processing
        """
        self.n_cores = n_cores
    
    def jaccard_index(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Jaccard index between two binary vectors.
        
        Parameters
        ----------
        vec1, vec2 : np.ndarray
            Input vectors
            
        Returns
        -------
        float
            Jaccard index (intersection / union)
        """
        vec1_bin = (vec1 > 0).astype(int)
        vec2_bin = (vec2 > 0).astype(int)
        
        intersection = np.sum(vec1_bin * vec2_bin)
        union = np.sum((vec1_bin + vec2_bin) > 0)
        
        return safe_divide(intersection, union, fill_value=0.0)
    
    def pairwise_jaccard(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise Jaccard indices for all columns in matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix where each column is a firm and each row is a sector
            
        Returns
        -------
        np.ndarray
            Pairwise Jaccard similarity matrix
        """
        binary_matrix = (matrix > 0).astype(int)
        n_firms = matrix.shape[1]
        
        # Calculate intersection matrix: binary_matrix.T @ binary_matrix
        intersections = binary_matrix.T @ binary_matrix
        
        # Calculate column sums (number of sectors each firm connects to)
        column_sums = np.sum(binary_matrix, axis=0)
        
        # Calculate union matrix: outer sum - intersection
        unions = np.add.outer(column_sums, column_sums) - intersections
        
        # Calculate Jaccard matrix
        jaccard_matrix = safe_divide(intersections, unions, fill_value=0.0)
        
        return jaccard_matrix
    
    @optimize_matrix_operations
    def pairwise_weighted_jaccard(self, matrix: np.ndarray, use_sparse: bool = False) -> np.ndarray:
        """
        Calculate pairwise weighted Jaccard indices using optimized vectorized operations.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix where each column is a firm
        use_sparse : bool, default False
            Whether to use sparse matrix operations
            
        Returns
        -------
        np.ndarray
            Pairwise weighted Jaccard similarity matrix
        """
        n_firms = matrix.shape[1]
        
        # Check memory requirements for large matrices
        memory_required = estimate_memory_usage((matrix.shape[0], n_firms, n_firms))
        
        if memory_required > 1000:  # > 1GB
            warnings.warn(f"Large memory requirement ({memory_required:.1f} MB). Consider using smaller batch sizes.")
        
        # For very large matrices, process in blocks
        if n_firms > 1000 and memory_required > 500:
            return self._pairwise_weighted_jaccard_blocked(matrix, block_size=500)
        
        # Vectorized computation using broadcasting
        # Expand matrix to enable broadcasting: (n_sectors, n_firms, 1) and (n_sectors, 1, n_firms)
        matrix_i = matrix[:, :, np.newaxis]  # Shape: (n_sectors, n_firms, 1)
        matrix_j = matrix[:, np.newaxis, :]  # Shape: (n_sectors, 1, n_firms)
        
        # Use batch processing for memory efficiency if needed
        if memory_required > 100:  # > 100MB
            minimums, maximums = batch_minimum_maximum(matrix_i, matrix_j, batch_size=200)
        else:
            minimums = np.minimum(matrix_i, matrix_j)
            maximums = np.maximum(matrix_i, matrix_j)
        
        # Compute intersections and unions for all pairs simultaneously
        intersections = np.sum(minimums, axis=0)  # Shape: (n_firms, n_firms)
        unions = np.sum(maximums, axis=0)         # Shape: (n_firms, n_firms)
        
        # Calculate Jaccard indices with safe division
        result = safe_divide(intersections, unions, fill_value=0.0)
        
        return result
    
    def _pairwise_weighted_jaccard_blocked(self, matrix: np.ndarray, block_size: int = 500) -> np.ndarray:
        """
        Calculate pairwise weighted Jaccard indices using block processing for large matrices.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix where each column is a firm
        block_size : int
            Size of blocks for processing
            
        Returns
        -------
        np.ndarray
            Pairwise weighted Jaccard similarity matrix
        """
        n_firms = matrix.shape[1]
        result = np.zeros((n_firms, n_firms))
        
        for i in range(0, n_firms, block_size):
            end_i = min(i + block_size, n_firms)
            for j in range(0, n_firms, block_size):
                end_j = min(j + block_size, n_firms)
                
                # Process block
                block_i = matrix[:, i:end_i]
                block_j = matrix[:, j:end_j]
                
                # Vectorized computation for this block
                matrix_i = block_i[:, :, np.newaxis]
                matrix_j = block_j[:, np.newaxis, :]
                
                intersections = np.sum(np.minimum(matrix_i, matrix_j), axis=0)
                unions = np.sum(np.maximum(matrix_i, matrix_j), axis=0)
                
                result[i:end_i, j:end_j] = safe_divide(intersections, unions, fill_value=0.0)
        
        return result
    
    def pairwise_weighted_jaccard_relative(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise relative weighted Jaccard indices.
        
        Normalizes each column by its sum before calculating weighted Jaccard.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix where each column is a firm
            
        Returns
        -------
        np.ndarray
            Pairwise relative weighted Jaccard similarity matrix
        """
        # Normalize columns by their sums
        column_sums = np.sum(matrix, axis=0)
        normalized_matrix = safe_divide(matrix, column_sums[np.newaxis, :], fill_value=0.0)
        
        return self.pairwise_weighted_jaccard(normalized_matrix)
    
    @optimize_matrix_operations
    def pairwise_overlap_relative(self, matrix: np.ndarray, use_sparse: bool = False) -> np.ndarray:
        """
        Calculate pairwise relative overlap coefficients using optimized vectorized operations.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix where each column is a firm
        use_sparse : bool, default False
            Whether to use sparse matrix operations
            
        Returns
        -------
        np.ndarray
            Pairwise relative overlap matrix
        """
        # Normalize columns by their sums
        column_sums = np.sum(matrix, axis=0)
        normalized_matrix = safe_divide(matrix, column_sums[np.newaxis, :], fill_value=0.0)
        
        n_firms = matrix.shape[1]
        
        # Check memory requirements
        memory_required = estimate_memory_usage((matrix.shape[0], n_firms, n_firms))
        
        if memory_required > 500:  # > 500MB, use blocked processing
            return self._pairwise_overlap_blocked(normalized_matrix, block_size=400)
        
        # Vectorized computation using broadcasting
        # Expand matrix to enable broadcasting: (n_sectors, n_firms, 1) and (n_sectors, 1, n_firms)
        matrix_i = normalized_matrix[:, :, np.newaxis]  # Shape: (n_sectors, n_firms, 1)
        matrix_j = normalized_matrix[:, np.newaxis, :]  # Shape: (n_sectors, 1, n_firms)
        
        # Use batch processing for memory efficiency if needed
        if memory_required > 100:  # > 100MB
            minimums, _ = batch_minimum_maximum(matrix_i, matrix_j, batch_size=200)
        else:
            minimums = np.minimum(matrix_i, matrix_j)
        
        # Compute overlaps for all pairs simultaneously
        overlaps = np.sum(minimums, axis=0)  # Shape: (n_firms, n_firms)
        
        return overlaps
    
    def _pairwise_overlap_blocked(self, normalized_matrix: np.ndarray, block_size: int = 400) -> np.ndarray:
        """
        Calculate pairwise overlap using block processing for large matrices.
        
        Parameters
        ----------
        normalized_matrix : np.ndarray
            Normalized input matrix where each column is a firm
        block_size : int
            Size of blocks for processing
            
        Returns
        -------
        np.ndarray
            Pairwise overlap matrix
        """
        n_firms = normalized_matrix.shape[1]
        result = np.zeros((n_firms, n_firms))
        
        for i in range(0, n_firms, block_size):
            end_i = min(i + block_size, n_firms)
            for j in range(0, n_firms, block_size):
                end_j = min(j + block_size, n_firms)
                
                # Process block
                block_i = normalized_matrix[:, i:end_i]
                block_j = normalized_matrix[:, j:end_j]
                
                # Vectorized computation for this block
                matrix_i = block_i[:, :, np.newaxis]
                matrix_j = block_j[:, np.newaxis, :]
                
                overlaps = np.sum(np.minimum(matrix_i, matrix_j), axis=0)
                result[i:end_i, j:end_j] = overlaps
        
        return result
    
    def pairwise_cosine_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise cosine similarities.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix where each column is a firm
            
        Returns
        -------
        np.ndarray
            Pairwise cosine similarity matrix
        """
        # Calculate dot products
        dot_products = matrix.T @ matrix
        
        # Calculate norms
        norms = np.sqrt(np.sum(matrix**2, axis=0))
        
        # Calculate cosine similarities
        norm_products = np.outer(norms, norms)
        cosine_matrix = safe_divide(dot_products, norm_products, fill_value=0.0)
        
        return cosine_matrix
    
    def pairwise_link_retention(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise link retention rates.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix where each column is a firm
            
        Returns
        -------
        np.ndarray
            Pairwise link retention matrix
        """
        binary_matrix = (matrix > 0).astype(int)
        
        # Calculate intersections
        intersections = binary_matrix.T @ binary_matrix
        
        # Calculate column sums
        column_sums = np.sum(binary_matrix, axis=0)
        
        # Calculate retention rates
        retention_matrix = safe_divide(intersections, column_sums[:, np.newaxis], fill_value=0.0)
        
        return retention_matrix
    
    def calculate_pairwise_similarity(self, matrix: np.ndarray, 
                                    similarity: str = "jaccard") -> np.ndarray:
        """
        Calculate pairwise similarity matrix using specified measure.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix where each column is a firm
        similarity : str, default "jaccard"
            Similarity measure: "jaccard", "weighted_jaccard", "weighted_jaccard_relative",
            "overlap_relative", "cosine", "retention"
            
        Returns
        -------
        np.ndarray
            Pairwise similarity matrix (lower triangular)
        """
        if matrix.shape[1] <= 1:
            return np.zeros((matrix.shape[1], matrix.shape[1]))
        
        if similarity == "jaccard":
            sim_matrix = self.pairwise_jaccard(matrix)
        elif similarity == "weighted_jaccard":
            sim_matrix = self.pairwise_weighted_jaccard(matrix)
        elif similarity == "weighted_jaccard_relative":
            sim_matrix = self.pairwise_weighted_jaccard_relative(matrix)
        elif similarity == "overlap_relative":
            sim_matrix = self.pairwise_overlap_relative(matrix)
        elif similarity == "cosine":
            sim_matrix = self.pairwise_cosine_similarity(matrix)
        elif similarity == "retention":
            sim_matrix = self.pairwise_link_retention(matrix)
        else:
            raise ValueError(f"Unknown similarity measure: {similarity}")
        
        # Return lower triangular matrix
        return np.tril(sim_matrix)
    
    def temporal_jaccard(self, matrix_t: np.ndarray, matrix_tm1: np.ndarray) -> np.ndarray:
        """
        Calculate temporal Jaccard indices between time periods.
        
        Parameters
        ----------
        matrix_t : np.ndarray
            Matrix at time t
        matrix_tm1 : np.ndarray
            Matrix at time t-1
            
        Returns
        -------
        np.ndarray
            Temporal Jaccard indices for each firm
        """
        binary_t = (matrix_t > 0).astype(int)
        binary_tm1 = (matrix_tm1 > 0).astype(int)
        
        intersections = np.sum(binary_t * binary_tm1, axis=0)
        unions = np.sum((binary_t + binary_tm1) > 0, axis=0)
        
        return safe_divide(intersections, unions, fill_value=0.0)
    
    def temporal_link_retention(self, matrix_t: np.ndarray, matrix_tm1: np.ndarray) -> np.ndarray:
        """
        Calculate temporal link retention rates.
        
        Parameters
        ----------
        matrix_t : np.ndarray
            Matrix at time t
        matrix_tm1 : np.ndarray
            Matrix at time t-1
            
        Returns
        -------
        np.ndarray
            Link retention rates for each firm
        """
        binary_t = (matrix_t > 0).astype(int)
        binary_tm1 = (matrix_tm1 > 0).astype(int)
        
        intersections = np.sum(binary_t * binary_tm1, axis=0)
        tm1_sums = np.sum(binary_tm1, axis=0)
        
        return safe_divide(intersections, tm1_sums, fill_value=0.0)
    
    def temporal_weighted_jaccard(self, matrix_t: np.ndarray, matrix_tm1: np.ndarray) -> np.ndarray:
        """
        Calculate temporal weighted Jaccard indices.
        
        Parameters
        ----------
        matrix_t : np.ndarray
            Matrix at time t
        matrix_tm1 : np.ndarray
            Matrix at time t-1
            
        Returns
        -------
        np.ndarray
            Temporal weighted Jaccard indices for each firm
        """
        intersections = np.sum(np.minimum(matrix_t, matrix_tm1), axis=0)
        unions = np.sum(np.maximum(matrix_t, matrix_tm1), axis=0)
        
        return safe_divide(intersections, unions, fill_value=0.0)
    
    def temporal_cosine_similarity(self, matrix_t: np.ndarray, matrix_tm1: np.ndarray) -> np.ndarray:
        """
        Calculate temporal cosine similarities.
        
        Parameters
        ----------
        matrix_t : np.ndarray
            Matrix at time t
        matrix_tm1 : np.ndarray
            Matrix at time t-1
            
        Returns
        -------
        np.ndarray
            Temporal cosine similarities for each firm
        """
        dot_products = np.sum(matrix_t * matrix_tm1, axis=0)
        norms_t = np.sqrt(np.sum(matrix_t**2, axis=0))
        norms_tm1 = np.sqrt(np.sum(matrix_tm1**2, axis=0))
        
        norm_products = norms_t * norms_tm1
        
        return safe_divide(dot_products, norm_products, fill_value=0.0)
    
    def calculate_temporal_similarity(self, matrix_t: np.ndarray, matrix_tm1: np.ndarray,
                                    similarity: str = "jaccard") -> np.ndarray:
        """
        Calculate temporal similarity using specified measure.
        
        Parameters
        ----------
        matrix_t : np.ndarray
            Matrix at time t
        matrix_tm1 : np.ndarray
            Matrix at time t-1
        similarity : str, default "jaccard"
            Similarity measure
            
        Returns
        -------
        np.ndarray
            Temporal similarities for each firm
        """
        if similarity == "jaccard":
            return self.temporal_jaccard(matrix_t, matrix_tm1)
        elif similarity == "weighted_jaccard":
            return self.temporal_weighted_jaccard(matrix_t, matrix_tm1)
        elif similarity == "cosine":
            return self.temporal_cosine_similarity(matrix_t, matrix_tm1)
        elif similarity == "retention":
            return self.temporal_link_retention(matrix_t, matrix_tm1)
        else:
            raise ValueError(f"Unknown similarity measure: {similarity}")
    
    def calculate_io_similarities(self, network: np.ndarray, sectors: Union[List, np.ndarray],
                                direction: str = "input", measure: str = "jaccard",
                                input_threshold: tuple = (0, np.inf)) -> Dict[str, np.ndarray]:
        """
        Calculate input/output vector similarities for firms within sectors.
        
        Parameters
        ----------
        network : np.ndarray
            Firm-level adjacency matrix
        sectors : array-like
            Sector affiliation for each firm
        direction : str, default "input"
            Direction to analyze: "input" or "output"
        measure : str, default "jaccard"
            Similarity measure to use
        input_threshold : tuple, default (0, inf)
            Threshold for firm connectivity
            
        Returns
        -------
        dict
            Dictionary with sector names as keys and similarity matrices as values
        """
        from .network_aggregation import NetworkAggregator
        
        aggregator = NetworkAggregator()
        
        if direction == "input":
            agg_data = aggregator.aggregate_suppliers(network, sectors)
        elif direction == "output":
            agg_data = aggregator.aggregate_buyers(network, sectors)
        else:
            raise ValueError("Direction must be 'input' or 'output'")
        
        sector_matrices = aggregator.calculate_sector_matrices(
            network, sectors, input_threshold=input_threshold
        )
        
        similarities = {}
        for sector_name, sector_matrix in sector_matrices.items():
            if sector_matrix.shape[1] > 0:
                similarities[sector_name] = self.calculate_pairwise_similarity(
                    sector_matrix, similarity=measure
                )
            else:
                similarities[sector_name] = np.array([[]])
        
        return similarities