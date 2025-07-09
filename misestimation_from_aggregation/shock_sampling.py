"""
Synthetic shock sampling algorithms for maintaining sector-level consistency.
"""

import numpy as np
from typing import Union, List, Dict, Optional, Tuple
import warnings
from scipy.linalg import pinv
from .utils import validate_network, validate_sectors, safe_divide


class ShockSampler:
    """
    Class for sampling synthetic firm-level shocks that maintain sector-level consistency.
    
    Implements the algorithm from the paper for generating heterogeneous firm-level
    shocks that aggregate to the same sector-level impact.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize shock sampler.
        
        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducible results
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self.random_seed = random_seed
    
    def sample_firm_level_shocks(self, firm_shock: Optional[np.ndarray] = None,
                                sector_shocks: Optional[Dict[str, Tuple[float, float]]] = None,
                                network: np.ndarray = None,
                                sector_affiliations: Union[List, np.ndarray] = None,
                                n_scenarios: int = 10,
                                sample_mode: str = "empirical",
                                tracker: bool = False,
                                silent: bool = True) -> Union[np.ndarray, Dict]:
        """
        Sample synthetic firm-level shocks maintaining sector-level consistency.
        
        Parameters
        ----------
        firm_shock : np.ndarray, optional
            Empirical firm-level shock vector (values in [0,1])
        sector_shocks : dict, optional
            Dictionary with sector names as keys and (in_shock, out_shock) tuples as values
        network : np.ndarray
            Firm-level adjacency matrix
        sector_affiliations : array-like
            Sector affiliation for each firm
        n_scenarios : int, default 10
            Number of shock scenarios to generate
        sample_mode : str, default "empirical"
            Sampling mode: "empirical" or "uniform"
        tracker : bool, default False
            Whether to track convergence statistics
        silent : bool, default True
            Whether to suppress progress output
            
        Returns
        -------
        np.ndarray or dict
            Shock matrix (n_firms, n_scenarios) or dict with shock matrix and tracking info
        """
        network = validate_network(network)
        sector_affiliations = validate_sectors(sector_affiliations, network.shape[0])
        n_firms = network.shape[0]
        
        # Determine which shock specification to use
        if firm_shock is not None and sector_shocks is not None:
            if not silent:
                print("Warning: Both firm-level and sector-level shocks provided. Using firm-level shocks.")
            sector_shocks = None
        
        if firm_shock is not None:
            # Calculate sector-level shocks from firm-level empirical data
            sector_shocks = self._calculate_sector_shocks_from_firm_data(
                firm_shock, network, sector_affiliations
            )
        elif sector_shocks is None:
            raise ValueError("Either firm_shock or sector_shocks must be provided")
        
        # Calculate firm strengths
        s_in = np.sum(network, axis=0)   # in-strength (column sums)
        s_out = np.sum(network, axis=1)  # out-strength (row sums)
        
        # Get unique sectors and initialize outputs
        unique_sectors = np.unique(sector_affiliations)
        n_sectors = len(unique_sectors)
        
        shock_indices = []
        tracking_data = []
        
        if not silent:
            print(f"{n_sectors} industry sectors contained in sector_affiliations")
        
        # Process each sector
        for i, sector in enumerate(unique_sectors):
            sector_mask = sector_affiliations == sector
            sector_firm_ids = np.where(sector_mask)[0]
            n_sector_firms = len(sector_firm_ids)
            
            if not silent:
                print(f"SECTOR: {i+1} out of {n_sectors} | sector name: {sector} | sector size: {n_sector_firms}")
            
            # Get sector shock parameters
            if str(sector) in sector_shocks:
                psi_k_in, psi_k_out = sector_shocks[str(sector)]
            else:
                # Skip sectors with no shock
                continue
            
            # Sample empirical shock distribution for this sector
            if firm_shock is not None:
                sector_firm_shocks = firm_shock[sector_mask]
                emp_shock_dist = sector_firm_shocks[sector_firm_shocks > 0]
            else:
                emp_shock_dist = np.array([])
            
            # Sample shocks for this sector
            sector_shock_result = self._sample_sector_shocks(
                psi_k_in=psi_k_in,
                psi_k_out=psi_k_out,
                s_in=s_in[sector_firm_ids],
                s_out=s_out[sector_firm_ids],
                n_scenarios=n_scenarios,
                n_sector_firms=n_sector_firms,
                sector_firm_ids=sector_firm_ids,
                empirical_shock_dist=emp_shock_dist,
                sample_mode=sample_mode,
                tracker=tracker,
                silent=silent
            )
            
            shock_indices.append(sector_shock_result['indices'])
            if tracker:
                tracking_data.append(sector_shock_result['tracking'])
        
        # Create sparse shock matrix
        if shock_indices:
            all_indices = np.vstack(shock_indices)
            shock_matrix = np.zeros((n_firms, n_scenarios))
            shock_matrix[all_indices[:, 0].astype(int), all_indices[:, 1].astype(int)] = all_indices[:, 2]
        else:
            shock_matrix = np.zeros((n_firms, n_scenarios))
        
        if tracker:
            return {
                'shock_matrix': shock_matrix,
                'tracking_data': tracking_data
            }
        else:
            return shock_matrix
    
    def _calculate_sector_shocks_from_firm_data(self, firm_shock: np.ndarray,
                                               network: np.ndarray,
                                               sector_affiliations: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Calculate sector-level shock parameters from empirical firm-level data.
        
        Parameters
        ----------
        firm_shock : np.ndarray
            Empirical firm-level shock vector
        network : np.ndarray
            Firm-level adjacency matrix
        sector_affiliations : np.ndarray
            Sector affiliation for each firm
            
        Returns
        -------
        dict
            Dictionary with sector shock parameters
        """
        s_in = np.sum(network, axis=0)
        s_out = np.sum(network, axis=1)
        
        unique_sectors = np.unique(sector_affiliations)
        sector_shocks = {}
        
        for sector in unique_sectors:
            sector_mask = sector_affiliations == sector
            
            # Calculate sector totals
            sector_s_in_total = np.sum(s_in[sector_mask])
            sector_s_out_total = np.sum(s_out[sector_mask])
            
            # Calculate sector shock totals
            sector_s_in_shock = np.sum(s_in[sector_mask] * firm_shock[sector_mask])
            sector_s_out_shock = np.sum(s_out[sector_mask] * firm_shock[sector_mask])
            
            # Calculate relative shocks
            if sector_s_in_total > 1e-10:
                psi_k_in = sector_s_in_shock / sector_s_in_total
            else:
                psi_k_in = 0.0
            
            if sector_s_out_total > 1e-10:
                psi_k_out = sector_s_out_shock / sector_s_out_total
            else:
                psi_k_out = 0.0
            
            sector_shocks[str(sector)] = (psi_k_in, psi_k_out)
        
        return sector_shocks
    
    def _sample_sector_shocks(self, psi_k_in: float, psi_k_out: float,
                             s_in: np.ndarray, s_out: np.ndarray,
                             n_scenarios: int, n_sector_firms: int,
                             sector_firm_ids: np.ndarray,
                             empirical_shock_dist: np.ndarray,
                             sample_mode: str = "empirical",
                             tracker: bool = False,
                             silent: bool = True) -> Dict:
        """
        Sample shocks for a single sector.
        
        Parameters
        ----------
        psi_k_in, psi_k_out : float
            Target sector shock percentages for in/out strength
        s_in, s_out : np.ndarray
            In/out strengths of firms in this sector
        n_scenarios : int
            Number of scenarios to generate
        n_sector_firms : int
            Number of firms in sector
        sector_firm_ids : np.ndarray
            Original firm IDs in this sector
        empirical_shock_dist : np.ndarray
            Empirical shock distribution to sample from
        sample_mode : str
            Sampling mode: "empirical" or "uniform"
        tracker : bool
            Whether to track convergence
        silent : bool
            Whether to suppress output
            
        Returns
        -------
        dict
            Dictionary with shock indices and tracking data
        """
        # Handle edge cases
        if n_sector_firms == 1:
            # Single firm gets average shock
            avg_shock = np.mean([psi_k_in, psi_k_out])
            indices = np.array([[sector_firm_ids[0], s, avg_shock] for s in range(n_scenarios)])
            return {'indices': indices, 'tracking': None}
        
        if np.mean([psi_k_in, psi_k_out]) <= 1e-15:
            # Zero shock
            indices = np.empty((0, 3))
            return {'indices': indices, 'tracking': None}
        
        # Calculate absolute shock targets
        s_in_shock_target = np.sum(s_in) * psi_k_in
        s_out_shock_target = np.sum(s_out) * psi_k_out
        
        # Initialize tracking
        if tracker:
            tracking_matrix = np.zeros((n_scenarios, 3))
        
        # Sample initial shocks to reach targets
        shock_matrix = self._initial_shock_sampling(
            s_in, s_out, s_in_shock_target, s_out_shock_target,
            n_scenarios, n_sector_firms, empirical_shock_dist, sample_mode
        )
        
        # Refine shocks using constraint satisfaction
        final_shocks = np.zeros((n_sector_firms, n_scenarios))
        
        for scenario in range(n_scenarios):
            refined_shock = self._refine_sector_shock(
                shock_matrix[:, scenario], s_in, s_out,
                s_in_shock_target, s_out_shock_target,
                silent=silent
            )
            final_shocks[:, scenario] = refined_shock
            
            if tracker:
                # Calculate convergence metrics
                final_s_in = np.sum(s_in * refined_shock)
                final_s_out = np.sum(s_out * refined_shock)
                in_error = abs(final_s_in - s_in_shock_target)
                out_error = abs(final_s_out - s_out_shock_target)
                tracking_matrix[scenario, :] = [0, in_error, out_error]  # iterations set to 0 for now
        
        # Convert to sparse indices format
        rows, cols = np.where(final_shocks > 0)
        values = final_shocks[rows, cols]
        
        # Map back to original firm indices
        original_rows = sector_firm_ids[rows]
        indices = np.column_stack([original_rows.astype(int), cols.astype(int), values])
        
        result = {'indices': indices}
        if tracker:
            result['tracking'] = tracking_matrix
        
        return result
    
    def _initial_shock_sampling(self, s_in: np.ndarray, s_out: np.ndarray,
                               s_in_target: float, s_out_target: float,
                               n_scenarios: int, n_firms: int,
                               empirical_dist: np.ndarray, sample_mode: str) -> np.ndarray:
        """
        Generate initial shock samples that approximately reach targets.
        
        Parameters
        ----------
        s_in, s_out : np.ndarray
            Firm strengths
        s_in_target, s_out_target : float
            Target shock levels
        n_scenarios : int
            Number of scenarios
        n_firms : int
            Number of firms
        empirical_dist : np.ndarray
            Empirical distribution to sample from
        sample_mode : str
            Sampling mode
            
        Returns
        -------
        np.ndarray
            Initial shock matrix
        """
        shock_matrix = np.zeros((n_firms, n_scenarios))
        
        for scenario in range(n_scenarios):
            # Permute firm order for this scenario
            perm_order = np.random.permutation(n_firms)
            inv_perm = np.argsort(perm_order)
            
            s_in_perm = s_in[perm_order]
            s_out_perm = s_out[perm_order]
            
            current_shocks = np.zeros(n_firms)
            current_s_in = 0
            current_s_out = 0
            
            # Add shocks until targets are reached
            for i in range(n_firms):
                if current_s_in >= s_in_target and current_s_out >= s_out_target:
                    break
                
                # Sample shock
                if sample_mode == "empirical" and len(empirical_dist) > 0:
                    shock = np.random.choice(empirical_dist)
                else:
                    shock = np.random.uniform(0, 1)
                
                current_shocks[i] = min(1.0, shock)
                current_s_in += s_in_perm[i] * current_shocks[i]
                current_s_out += s_out_perm[i] * current_shocks[i]
            
            # Permute back to original order
            shock_matrix[:, scenario] = current_shocks[inv_perm]
        
        return shock_matrix
    
    def _refine_sector_shock(self, initial_shock: np.ndarray, s_in: np.ndarray, s_out: np.ndarray,
                           s_in_target: float, s_out_target: float,
                           max_iterations: int = 1000, tolerance: float = 1e-2,
                           silent: bool = True) -> np.ndarray:
        """
        Refine shock vector to exactly meet sector targets using constraint satisfaction.
        
        Parameters
        ----------
        initial_shock : np.ndarray
            Initial shock vector
        s_in, s_out : np.ndarray
            Firm strengths
        s_in_target, s_out_target : float
            Target shock levels
        max_iterations : int
            Maximum refinement iterations
        tolerance : float
            Convergence tolerance
        silent : bool
            Whether to suppress output
            
        Returns
        -------
        np.ndarray
            Refined shock vector
        """
        shock = initial_shock.copy()
        n_firms = len(shock)
        
        # Determine firm types based on strength ratios
        strength_ratio = safe_divide(s_in, s_out, fill_value=1.0)
        target_ratio = safe_divide(s_in_target, s_out_target, fill_value=1.0)
        
        in_heavy = strength_ratio > target_ratio
        out_heavy = ~in_heavy
        
        # Handle edge cases
        if not np.any(in_heavy):
            in_heavy[np.argmax(strength_ratio)] = True
            out_heavy[np.argmax(strength_ratio)] = False
        
        if not np.any(out_heavy):
            out_heavy[np.argmin(strength_ratio)] = True
            in_heavy[np.argmin(strength_ratio)] = False
        
        full_shock_firms = np.zeros(n_firms, dtype=bool)
        
        for iteration in range(max_iterations):
            # Calculate remaining targets
            remaining_s_in_target = s_in_target - np.sum(full_shock_firms * s_in)
            remaining_s_out_target = s_out_target - np.sum(full_shock_firms * s_out)
            
            if remaining_s_in_target <= tolerance and remaining_s_out_target <= tolerance:
                break
            
            # Update firm classifications
            available_firms = ~full_shock_firms
            if not np.any(available_firms):
                break
            
            # Set up constraint system: A * rescale_factors = targets
            # where rescale_factors are for [in_heavy, out_heavy] groups
            in_heavy_contrib = np.sum((s_in * shock * available_firms)[in_heavy & available_firms])
            out_heavy_contrib = np.sum((s_out * shock * available_firms)[out_heavy & available_firms])
            in_heavy_out_contrib = np.sum((s_out * shock * available_firms)[in_heavy & available_firms])
            out_heavy_in_contrib = np.sum((s_in * shock * available_firms)[out_heavy & available_firms])
            
            A = np.array([[in_heavy_contrib, out_heavy_in_contrib],
                         [in_heavy_out_contrib, out_heavy_contrib]])
            b = np.array([remaining_s_in_target, remaining_s_out_target])
            
            # Solve for rescaling factors
            try:
                if np.linalg.det(A) != 0:
                    rescale_factors = np.linalg.solve(A, b)
                else:
                    rescale_factors = pinv(A) @ b
                
                # Ensure non-negative rescaling
                rescale_factors = np.maximum(0, rescale_factors)
                
                # Apply rescaling
                new_shock = shock.copy()
                new_shock[in_heavy & available_firms] *= rescale_factors[0]
                new_shock[out_heavy & available_firms] *= rescale_factors[1]
                
                # Cap at 1.0 and mark fully shocked firms
                new_shock = np.minimum(1.0, new_shock)
                newly_full = (new_shock >= 1.0) & ~full_shock_firms
                full_shock_firms |= newly_full
                shock = new_shock
                
            except np.linalg.LinAlgError:
                # If linear system fails, use simple proportional scaling
                current_s_in = np.sum(s_in * shock)
                current_s_out = np.sum(s_out * shock)
                
                if current_s_in > 0:
                    in_scale = s_in_target / current_s_in
                else:
                    in_scale = 1.0
                
                if current_s_out > 0:
                    out_scale = s_out_target / current_s_out
                else:
                    out_scale = 1.0
                
                avg_scale = np.mean([in_scale, out_scale])
                shock *= avg_scale
                shock = np.minimum(1.0, shock)
                break
        
        return shock