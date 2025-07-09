"""
Network aggregation functions for converting firm-level networks to sector-level.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from .utils import validate_network, validate_sectors, get_sector_mapping


class NetworkAggregator:
    """
    Class for aggregating firm-level networks to sector-level representations.
    
    This class provides methods to aggregate supply networks by grouping firms
    into sectors and computing sector-to-sector connections.
    """
    
    def __init__(self):
        pass
    
    def aggregate_suppliers(self, network: np.ndarray, sectors: Union[List, np.ndarray],
                           unique_sectors: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Aggregate firm-level supply network to sectors on the supplier side.
        
        Creates a sectors × firms matrix where element (k,i) represents the amount
        firm i bought from sector k.
        
        Parameters
        ----------
        network : np.ndarray
            Firm-level adjacency matrix, shape (n_firms, n_firms)
        sectors : array-like
            Sector affiliation for each firm
        unique_sectors : np.ndarray, optional
            Predefined list of sectors to include
            
        Returns
        -------
        dict
            Dictionary with 'counts' and 'volume' keys containing aggregated matrices
        """
        network = validate_network(network)
        sectors = validate_sectors(sectors, network.shape[0])
        
        if unique_sectors is None:
            unique_sectors = np.unique(sectors)
        else:
            unique_sectors = np.asarray(unique_sectors)
        
        n_firms = network.shape[0]
        n_sectors = len(unique_sectors)
        
        # Create indicator matrix: psup[i, k] = 1 if firm i is in sector k
        psup = np.zeros((n_firms, n_sectors))
        for i, sector in enumerate(unique_sectors):
            psup[:, i] = (sectors == sector).astype(int)
        
        # Aggregate: suppliers become sectors
        # counts: binary network (connections)
        # volume: weighted network (transaction volumes)
        binary_network = (network > 0).astype(int)
        
        return {
            'counts': psup.T @ binary_network,  # (n_sectors, n_firms)
            'volume': psup.T @ network          # (n_sectors, n_firms)
        }
    
    def aggregate_buyers(self, network: np.ndarray, sectors: Union[List, np.ndarray],
                        unique_sectors: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Aggregate firm-level supply network to sectors on the buyer side.
        
        Creates a sectors × firms matrix where element (k,i) represents the amount
        firm i sold to sector k.
        
        Parameters
        ----------
        network : np.ndarray
            Firm-level adjacency matrix, shape (n_firms, n_firms)
        sectors : array-like
            Sector affiliation for each firm
        unique_sectors : np.ndarray, optional
            Predefined list of sectors to include
            
        Returns
        -------
        dict
            Dictionary with 'counts' and 'volume' keys containing aggregated matrices
        """
        network = validate_network(network)
        sectors = validate_sectors(sectors, network.shape[0])
        
        if unique_sectors is None:
            unique_sectors = np.unique(sectors)
        else:
            unique_sectors = np.asarray(unique_sectors)
        
        n_firms = network.shape[0]
        n_sectors = len(unique_sectors)
        
        # Create indicator matrix: psup[i, k] = 1 if firm i is in sector k
        psup = np.zeros((n_firms, n_sectors))
        for i, sector in enumerate(unique_sectors):
            psup[:, i] = (sectors == sector).astype(int)
        
        # Aggregate: buyers become sectors
        binary_network = (network > 0).astype(int)
        
        return {
            'counts': (binary_network @ psup).T,  # (n_sectors, n_firms)
            'volume': (network @ psup).T          # (n_sectors, n_firms)
        }
    
    def aggregate_to_sectors(self, network: np.ndarray, 
                           sectors: Union[List, np.ndarray]) -> np.ndarray:
        """
        Aggregate firm-level network to sector-level network.
        
        Parameters
        ----------
        network : np.ndarray
            Firm-level adjacency matrix, shape (n_firms, n_firms)
        sectors : array-like
            Sector affiliation for each firm
            
        Returns
        -------
        np.ndarray
            Sector-level adjacency matrix, shape (n_sectors, n_sectors)
        """
        network = validate_network(network)
        sectors = validate_sectors(sectors, network.shape[0])
        
        consecutive_sectors, sector_to_index, index_to_sector = get_sector_mapping(sectors)
        n_sectors = len(sector_to_index)
        n_firms = network.shape[0]
        
        # Create sector aggregation matrix
        psup = np.zeros((n_firms, n_sectors))
        for i in range(n_firms):
            psup[i, consecutive_sectors[i]] = 1
        
        # Aggregate: Z = P^T * W * P
        sector_network = psup.T @ network @ psup
        
        return sector_network
    
    def calculate_sector_matrices(self, network: np.ndarray, sectors: Union[List, np.ndarray],
                                 input_threshold: Tuple[float, float] = (0, np.inf),
                                 remove_na_sector: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Calculate sector-specific input/output matrices for firms within each sector.
        
        Parameters
        ----------
        network : np.ndarray
            Firm-level adjacency matrix
        sectors : array-like
            Sector affiliation for each firm
        input_threshold : tuple, default (0, inf)
            Min and max threshold for firm connectivity
        remove_na_sector : int, optional
            Sector ID to remove (e.g., for missing/NA sectors)
            
        Returns
        -------
        dict
            Dictionary with sector names as keys and input/output matrices as values
        """
        network = validate_network(network)
        sectors = validate_sectors(sectors, network.shape[0])
        
        # Aggregate suppliers and buyers
        supplier_agg = self.aggregate_suppliers(network, sectors)
        buyer_agg = self.aggregate_buyers(network, sectors)
        
        unique_sectors = np.unique(sectors)
        if remove_na_sector is not None:
            unique_sectors = unique_sectors[unique_sectors != remove_na_sector]
        
        sector_matrices = {}
        
        for sector in unique_sectors:
            sector_name = f"sector_{sector}"
            
            # Get firms in this sector
            sector_mask = sectors == sector
            sector_firms = np.where(sector_mask)[0]
            
            # Extract input/output matrices for this sector
            if remove_na_sector is not None:
                # Remove the NA sector row
                sector_idx = list(unique_sectors).index(sector)
                volume_matrix = supplier_agg['volume']
                volume_matrix = np.delete(volume_matrix, remove_na_sector, axis=0)
                
                counts_matrix = supplier_agg['counts']  
                counts_matrix = np.delete(counts_matrix, remove_na_sector, axis=0)
            else:
                volume_matrix = supplier_agg['volume'][:, sector_firms]
                counts_matrix = supplier_agg['counts'][:, sector_firms]
            
            # Apply input thresholds
            if len(sector_firms) > 0:
                firm_degrees = np.sum(counts_matrix, axis=0)
                threshold_mask = ((firm_degrees >= input_threshold[0]) & 
                                (firm_degrees <= input_threshold[1]))
                
                if np.any(threshold_mask):
                    sector_matrices[sector_name] = volume_matrix[:, threshold_mask]
                else:
                    sector_matrices[sector_name] = np.zeros((volume_matrix.shape[0], 0))
            else:
                sector_matrices[sector_name] = np.zeros((volume_matrix.shape[0], 0))
        
        return sector_matrices