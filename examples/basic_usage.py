#!/usr/bin/env python3
"""
Basic usage example for the misestimation_from_aggregation package.

This script demonstrates the core functionality by recreating the example 
from Figure 1 of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from misestimation_from_aggregation import (
    NetworkAggregator, 
    SimilarityCalculator, 
    ShockSampler
)


def main():
    """Run the basic example from Figure 1."""
    print("Misestimation from Aggregation - Basic Example")
    print("=" * 50)
    
    # 1. Create the network from Figure 1
    print("\n1. Creating network from Figure 1...")
    
    n_firms = 11
    sector_affiliations = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])
    
    # Define network edges (supplier, buyer, weight)
    edges = [
        (3, 6, 1), (4, 6, 1), (4, 7, 1), (5, 7, 1),
        (6, 1, 1), (6, 2, 1), (7, 10, 1), (7, 11, 1),
        (8, 1, 1), (8, 2, 1), (9, 11, 1)
    ]
    
    # Create adjacency matrix
    W = np.zeros((n_firms, n_firms))
    for supplier, buyer, weight in edges:
        W[supplier-1, buyer-1] = weight  # Convert to 0-indexed
    
    print(f"   - Firm-level network: {W.shape}")
    print(f"   - Number of edges: {np.sum(W > 0)}")
    print(f"   - Total flow: {np.sum(W)}")
    print(f"   - Sectors: {np.unique(sector_affiliations)}")
    
    # 2. Aggregate to sector level
    print("\n2. Aggregating to sector level...")
    
    aggregator = NetworkAggregator()
    Z = aggregator.aggregate_to_sectors(W, sector_affiliations)
    
    print(f"   - Sector-level network: {Z.shape}")
    print(f"   - Flow preserved: {np.sum(Z) == np.sum(W)}")
    print(f"   - Sector network:\n{Z}")
    
    # 3. Calculate input-output similarities
    print("\n3. Calculating input-output similarities...")
    
    similarity_calc = SimilarityCalculator()
    
    input_similarities = similarity_calc.calculate_io_similarities(
        W, sector_affiliations, direction="input", measure="overlap_relative"
    )
    
    output_similarities = similarity_calc.calculate_io_similarities(
        W, sector_affiliations, direction="output", measure="overlap_relative"
    )
    
    print("   - Input vector overlaps by sector:")
    for sector, sim_matrix in input_similarities.items():
        if sim_matrix.size > 1:
            # Get lower triangular values (pairwise similarities)
            lower_tri = sim_matrix[np.tril_indices_from(sim_matrix, k=-1)]
            if len(lower_tri) > 0:
                print(f"     {sector}: {lower_tri[0]:.3f}")
    
    print("   - Output vector overlaps by sector:")
    for sector, sim_matrix in output_similarities.items():
        if sim_matrix.size > 1:
            lower_tri = sim_matrix[np.tril_indices_from(sim_matrix, k=-1)]
            if len(lower_tri) > 0:
                print(f"     {sector}: {lower_tri[0]:.3f}")
    
    # 4. Generate synthetic shocks
    print("\n4. Generating synthetic shocks...")
    
    # Create empirical shock: 100% shock to firm 3, 0% to all others
    empirical_shock = np.zeros(n_firms)
    empirical_shock[2] = 1.0  # Firm 3 (0-indexed)
    
    sampler = ShockSampler(random_seed=100)
    synthetic_shocks = sampler.sample_firm_level_shocks(
        firm_shock=empirical_shock,
        network=W,
        sector_affiliations=sector_affiliations,
        n_scenarios=10,
        sample_mode="empirical",
        silent=True
    )
    
    print(f"   - Generated synthetic shocks: {synthetic_shocks.shape}")
    print(f"   - Shocked firms per scenario: {np.sum(synthetic_shocks > 0, axis=0)}")
    
    # 5. Verify sector-level consistency
    print("\n5. Verifying sector-level consistency...")
    
    s_in = np.sum(W, axis=0)   # In-strength
    s_out = np.sum(W, axis=1)  # Out-strength
    
    # Calculate original sector 2 shocks
    sector_2_mask = sector_affiliations == 2
    orig_in_shock = np.sum(s_in[sector_2_mask] * empirical_shock[sector_2_mask])
    orig_out_shock = np.sum(s_out[sector_2_mask] * empirical_shock[sector_2_mask])
    
    print(f"   - Original sector 2 in-shock: {orig_in_shock:.3f}")
    print(f"   - Original sector 2 out-shock: {orig_out_shock:.3f}")
    
    # Check synthetic shocks
    print("   - Synthetic sector 2 shocks:")
    for i in range(min(5, synthetic_shocks.shape[1])):
        synth_shock = synthetic_shocks[:, i]
        synth_in = np.sum(s_in[sector_2_mask] * synth_shock[sector_2_mask])
        synth_out = np.sum(s_out[sector_2_mask] * synth_shock[sector_2_mask])
        print(f"     Scenario {i+1}: in={synth_in:.3f}, out={synth_out:.3f}")
    
    # 6. Economic impact analysis
    print("\n6. Economic impact analysis...")
    
    # Calculate firm-level influence using degree centrality (simplified)
    firm_degrees = np.sum(W > 0, axis=1) + np.sum(W > 0, axis=0)
    firm_influence = firm_degrees / np.sum(firm_degrees)
    
    # Calculate losses for each scenario
    firm_level_losses = []
    for scenario in range(synthetic_shocks.shape[1]):
        loss = np.dot(firm_influence, synthetic_shocks[:, scenario])
        firm_level_losses.append(loss)
    
    # Sector-level equivalent
    sector_degrees = np.sum(Z > 0, axis=1) + np.sum(Z > 0, axis=0)
    sector_influence = sector_degrees / np.sum(sector_degrees)
    
    # Map empirical shock to sector level
    sector_shock = np.zeros(len(np.unique(sector_affiliations)))
    sector_shock[1] = orig_out_shock / max(np.sum(s_out[sector_2_mask]), 1e-10)  # Sector 2 (0-indexed)
    
    sector_level_loss = np.dot(sector_influence, sector_shock)
    
    print(f"   - Mean firm-level loss: {np.mean(firm_level_losses):.6f}")
    print(f"   - Sector-level loss: {sector_level_loss:.6f}")
    print(f"   - Prediction error: {100 * abs(np.mean(firm_level_losses) - sector_level_loss) / abs(sector_level_loss):.2f}%")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("See examples/tutorial_and_experiments.ipynb for more detailed analysis.")


if __name__ == "__main__":
    main()