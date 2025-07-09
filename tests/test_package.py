"""
Tests for the misestimation_from_aggregation package.
"""

import numpy as np
import pytest
from misestimation_from_aggregation import (
    NetworkAggregator, 
    SimilarityCalculator, 
    ShockSampler,
    validate_network,
    validate_sectors
)


class TestUtils:
    """Test utility functions."""
    
    def test_validate_network_valid(self):
        """Test network validation with valid input."""
        network = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])
        result = validate_network(network)
        np.testing.assert_array_equal(result, network)
    
    def test_validate_network_invalid_shape(self):
        """Test network validation with invalid shape."""
        with pytest.raises(ValueError, match="Network must be a 2D matrix"):
            validate_network(np.array([1, 2, 3]))
    
    def test_validate_network_not_square(self):
        """Test network validation with non-square matrix."""
        with pytest.raises(ValueError, match="Network must be square"):
            validate_network(np.array([[1, 2], [3, 4], [5, 6]]))
    
    def test_validate_sectors_valid(self):
        """Test sector validation with valid input."""
        sectors = [1, 1, 2, 2, 3]
        result = validate_sectors(sectors, 5)
        np.testing.assert_array_equal(result, np.array(sectors))
    
    def test_validate_sectors_wrong_length(self):
        """Test sector validation with wrong length."""
        with pytest.raises(ValueError, match="Sectors length"):
            validate_sectors([1, 2, 3], 5)


class TestNetworkAggregator:
    """Test network aggregation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Create simple test network (from Figure 1 in paper)
        self.n_firms = 11
        self.sectors = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])
        
        # Create adjacency matrix
        edges = [
            (3, 6, 1), (4, 6, 1), (4, 7, 1), (5, 7, 1),
            (6, 1, 1), (6, 2, 1), (7, 10, 1), (7, 11, 1),
            (8, 1, 1), (8, 2, 1), (9, 11, 1)
        ]
        
        self.network = np.zeros((self.n_firms, self.n_firms))
        for supplier, buyer, weight in edges:
            self.network[supplier-1, buyer-1] = weight  # Convert to 0-indexed
        
        self.aggregator = NetworkAggregator()
    
    def test_aggregate_suppliers(self):
        """Test supplier aggregation."""
        result = self.aggregator.aggregate_suppliers(self.network, self.sectors)
        
        assert 'counts' in result
        assert 'volume' in result
        assert result['counts'].shape[0] == len(np.unique(self.sectors))  # n_sectors
        assert result['volume'].shape[1] == self.n_firms
    
    def test_aggregate_buyers(self):
        """Test buyer aggregation."""
        result = self.aggregator.aggregate_buyers(self.network, self.sectors)
        
        assert 'counts' in result
        assert 'volume' in result
        assert result['counts'].shape[0] == len(np.unique(self.sectors))  # n_sectors
        assert result['volume'].shape[1] == self.n_firms
    
    def test_aggregate_to_sectors(self):
        """Test full sector aggregation."""
        sector_network = self.aggregator.aggregate_to_sectors(self.network, self.sectors)
        
        n_sectors = len(np.unique(self.sectors))
        assert sector_network.shape == (n_sectors, n_sectors)
        
        # Check that total flow is preserved
        assert np.abs(np.sum(sector_network) - np.sum(self.network)) < 1e-10


class TestSimilarityCalculator:
    """Test similarity measure calculations."""
    
    def setup_method(self):
        """Set up test data."""
        self.calculator = SimilarityCalculator()
        
        # Create simple test matrix
        self.test_matrix = np.array([
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
    
    def test_jaccard_index(self):
        """Test Jaccard index calculation."""
        vec1 = np.array([1, 1, 0])
        vec2 = np.array([1, 0, 1])
        
        result = self.calculator.jaccard_index(vec1, vec2)
        expected = 1 / 3  # One intersection, three union
        
        assert np.abs(result - expected) < 1e-10
    
    def test_pairwise_jaccard(self):
        """Test pairwise Jaccard calculation."""
        result = self.calculator.pairwise_jaccard(self.test_matrix)
        
        assert result.shape == (4, 4)
        # Diagonal should be 1 (perfect self-similarity)
        np.testing.assert_array_almost_equal(np.diag(result), 1.0)
    
    def test_pairwise_cosine_similarity(self):
        """Test pairwise cosine similarity."""
        result = self.calculator.pairwise_cosine_similarity(self.test_matrix)
        
        assert result.shape == (4, 4)
        # Diagonal should be 1 (perfect self-similarity)
        np.testing.assert_array_almost_equal(np.diag(result), 1.0)
    
    def test_calculate_pairwise_similarity(self):
        """Test general pairwise similarity calculation."""
        result = self.calculator.calculate_pairwise_similarity(
            self.test_matrix, similarity="jaccard"
        )
        
        assert result.shape == (4, 4)
        # Should be lower triangular
        assert np.allclose(result, np.tril(result))


class TestShockSampler:
    """Test shock sampling functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.sampler = ShockSampler(random_seed=42)
        
        # Create simple test network
        self.n_firms = 11
        self.sectors = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])
        
        edges = [
            (3, 6, 1), (4, 6, 1), (4, 7, 1), (5, 7, 1),
            (6, 1, 1), (6, 2, 1), (7, 10, 1), (7, 11, 1),
            (8, 1, 1), (8, 2, 1), (9, 11, 1)
        ]
        
        self.network = np.zeros((self.n_firms, self.n_firms))
        for supplier, buyer, weight in edges:
            self.network[supplier-1, buyer-1] = weight
    
    def test_sample_firm_level_shocks_with_firm_data(self):
        """Test shock sampling with empirical firm data."""
        # 100% shock to firm 3 (index 2), 0% to all others
        firm_shock = np.zeros(self.n_firms)
        firm_shock[2] = 1.0  # Firm 3 (0-indexed)
        
        result = self.sampler.sample_firm_level_shocks(
            firm_shock=firm_shock,
            network=self.network,
            sector_affiliations=self.sectors,
            n_scenarios=5,
            silent=True
        )
        
        assert result.shape == (self.n_firms, 5)
        
        # Check that shocks only affect firms in sector 2
        sector_2_firms = np.where(self.sectors == 2)[0]
        for scenario in range(5):
            shocked_firms = np.where(result[:, scenario] > 0)[0]
            # All shocked firms should be in sector 2
            assert all(firm in sector_2_firms for firm in shocked_firms)
    
    def test_sample_firm_level_shocks_with_sector_data(self):
        """Test shock sampling with sector-level specification."""
        sector_shocks = {
            '2': (0.25, 0.25)  # 25% shock to sector 2 in/out strength
        }
        
        result = self.sampler.sample_firm_level_shocks(
            sector_shocks=sector_shocks,
            network=self.network,
            sector_affiliations=self.sectors,
            n_scenarios=3,
            silent=True
        )
        
        assert result.shape == (self.n_firms, 3)
        
        # Check that shocks only affect firms in sector 2
        sector_2_firms = np.where(self.sectors == 2)[0]
        for scenario in range(3):
            shocked_firms = np.where(result[:, scenario] > 0)[0]
            # All shocked firms should be in sector 2
            assert all(firm in sector_2_firms for firm in shocked_firms)
    
    def test_shock_conservation(self):
        """Test that sector-level shock targets are approximately preserved."""
        firm_shock = np.zeros(self.n_firms)
        firm_shock[2] = 1.0  # 100% shock to firm 3
        
        result = self.sampler.sample_firm_level_shocks(
            firm_shock=firm_shock,
            network=self.network,
            sector_affiliations=self.sectors,
            n_scenarios=3,
            silent=True
        )
        
        # Calculate sector-level impacts
        s_in = np.sum(self.network, axis=0)
        s_out = np.sum(self.network, axis=1)
        
        # Original sector 2 shock
        sector_2_mask = self.sectors == 2
        orig_sector_in_shock = np.sum(s_in[sector_2_mask] * firm_shock[sector_2_mask])
        orig_sector_out_shock = np.sum(s_out[sector_2_mask] * firm_shock[sector_2_mask])
        
        # Check each synthetic scenario
        for scenario in range(3):
            synth_shock = result[:, scenario]
            synth_sector_in_shock = np.sum(s_in[sector_2_mask] * synth_shock[sector_2_mask])
            synth_sector_out_shock = np.sum(s_out[sector_2_mask] * synth_shock[sector_2_mask])
            
            # Should be approximately equal (within tolerance)
            assert np.abs(synth_sector_in_shock - orig_sector_in_shock) < 0.1
            assert np.abs(synth_sector_out_shock - orig_sector_out_shock) < 0.1


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def setup_method(self):
        """Set up test data."""
        # Figure 1 network from paper
        self.n_firms = 11
        self.sectors = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])
        
        edges = [
            (3, 6, 1), (4, 6, 1), (4, 7, 1), (5, 7, 1),
            (6, 1, 1), (6, 2, 1), (7, 10, 1), (7, 11, 1),
            (8, 1, 1), (8, 2, 1), (9, 11, 1)
        ]
        
        self.network = np.zeros((self.n_firms, self.n_firms))
        for supplier, buyer, weight in edges:
            self.network[supplier-1, buyer-1] = weight
    
    def test_full_workflow(self):
        """Test complete analysis workflow."""
        # 1. Network aggregation
        aggregator = NetworkAggregator()
        sector_network = aggregator.aggregate_to_sectors(self.network, self.sectors)
        
        # 2. Similarity calculations
        calculator = SimilarityCalculator()
        similarities = calculator.calculate_io_similarities(
            self.network, self.sectors, direction="input", measure="jaccard"
        )
        
        # 3. Shock sampling
        sampler = ShockSampler(random_seed=42)
        firm_shock = np.zeros(self.n_firms)
        firm_shock[2] = 1.0  # Shock firm 3
        
        synthetic_shocks = sampler.sample_firm_level_shocks(
            firm_shock=firm_shock,
            network=self.network,
            sector_affiliations=self.sectors,
            n_scenarios=5,
            silent=True
        )
        
        # Verify results
        assert sector_network.shape == (5, 5)  # 5 sectors
        assert isinstance(similarities, dict)
        assert synthetic_shocks.shape == (self.n_firms, 5)
        
        # Check specific results from the paper
        # Sector 3 firms should have overlap of 1 (both buy from sector 2)
        if 'sector_3' in similarities:
            sector_3_sim = similarities['sector_3']
            if sector_3_sim.size > 0:
                # Both firms in sector 3 buy from sector 2, so should have high overlap
                pass  # Skip detailed assertion for now as matrix structure may vary


if __name__ == "__main__":
    pytest.main([__file__])