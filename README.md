# Misestimation from Aggregation

A Python package implementing the methodology from "Estimating the loss of economic predictability from aggregating firm-level production networks" by Diem, C., Borsos, A., Reisch, T., Kertész, J., & Thurner, S. (2023).

## Overview

This package provides tools for analyzing the misestimation that occurs when aggregating firm-level production networks to sector-level networks. It includes functions for:

- Calculating input-output vector overlaps (IOC and OOC similarities)
- Sampling synthetic firm-level shocks with controlled sector-level impacts
- Performing shock propagation analysis on production networks
- Comparing firm-level vs sector-level economic predictions

## Key Features

### Network Aggregation and Analysis
- Aggregate firm-level supply networks to sector-level representations
- Calculate various similarity measures (Jaccard, cosine, overlap coefficients)
- Support for both input and output vector analysis

### Shock Simulation
- Generate synthetic firm-level shocks that maintain sector-level consistency
- Sample from empirical shock distributions
- Control for heterogeneous firm-level impacts with homogeneous sector-level aggregates

### Economic Impact Assessment
- Shock propagation simulation using network-based models
- Influence vector calculations using PageRank-based approaches
- Comparison of firm-level vs sector-level predictive accuracy

## Installation

```bash
pip install misestimation_from_aggregation
```

For development installation:
```bash
git clone https://github.com/Basso42/misestimation_from_aggregation.git
cd misestimation_from_aggregation
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import networkx as nx
from misestimation_from_aggregation import (
    NetworkAggregator, 
    SimilarityCalculator, 
    ShockSampler
)

# Create a simple firm-level network (example from Figure 1)
n_firms = 11
sector_affiliations = [1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5]

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

# Aggregate to sector level
aggregator = NetworkAggregator()
sector_network = aggregator.aggregate_to_sectors(W, sector_affiliations)

# Calculate input-output similarities
similarity_calc = SimilarityCalculator()
input_similarities = similarity_calc.calculate_io_similarities(
    W, sector_affiliations, direction="input", measure="jaccard"
)

# Generate synthetic shocks
sampler = ShockSampler()
synthetic_shocks = sampler.sample_firm_level_shocks(
    firm_shock=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # 100% shock to firm 3
    network=W,
    sector_affiliations=sector_affiliations,
    n_scenarios=10
)

print(f"Generated {synthetic_shocks.shape[1]} synthetic shock scenarios")
```

## Mathematical Background

The package implements several key algorithms:

### Similarity Measures
- **Jaccard Index**: Measures overlap in binary input/output vectors
- **Weighted Jaccard**: Extension for weighted networks  
- **Cosine Similarity**: Measures vector angle similarity
- **Overlap Coefficient**: Relative overlap measure

### Shock Sampling Algorithm
The synthetic shock generation follows these steps:
1. Calculate sector-level shock targets from empirical firm-level data
2. Use iterative sampling to generate firm-level shocks
3. Apply constraint satisfaction to ensure sector-level consistency
4. Handle edge cases for sectors with few firms or extreme network structures

### Network Aggregation
Firms are grouped by sector affiliations, and connections are aggregated:
- **Input aggregation**: Sum inputs from each sector to each firm
- **Output aggregation**: Sum outputs from each firm to each sector
- **Sector network**: Create sector-to-sector adjacency matrix

## Core Modules

### `network_aggregation.py`
Functions for aggregating firm-level networks to sector-level representations and calculating input/output vectors.

### `similarity_measures.py` 
Implementation of various similarity measures for comparing input/output vectors between firms.

### `shock_sampling.py`
Algorithms for sampling synthetic firm-level shocks that maintain sector-level consistency.

### `utils.py`
Utility functions for data processing, validation, and mathematical operations.

## Documentation

Detailed documentation is available in the [docs/](docs/) directory and includes:
- API reference for all functions and classes
- Mathematical formulations and algorithms
- Extended examples and use cases
- Comparison with original R implementation

## Examples

See the [examples/](examples/) directory for:
- **Basic Usage**: Simple network analysis workflow
- **Replication Study**: Recreation of results from the original paper
- **Advanced Analysis**: Custom shock scenarios and network structures
- **Jupyter Notebook**: Interactive tutorial and experimentation

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=misestimation_from_aggregation tests/
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{diem2023estimating,
  title={Estimating the loss of economic predictability from aggregating firm-level production networks},
  author={Diem, Christoph and Borsos, András and Reisch, Tobias and Kertész, János and Thurner, Stefan},
  journal={},
  year={2023}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

For questions and support, please open an issue on the [GitHub repository](https://github.com/Basso42/misestimation_from_aggregation/issues).
