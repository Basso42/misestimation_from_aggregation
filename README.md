# misestimation_from_aggregation
Minimal replicateable example for Diem, C., Borsos, A., Reisch, T., Kertész, J., &amp; Thurner, S. (2023). Estimating the loss of economic predictability from aggregating firm-level production networks.

- misestimation_from_aggregation.R contains the code to produce all results from the paper for the example from Figure 1.
  
- GLcascade and fastcascade files contain the code to install the respective R packages used for the shock propagation simulation.
- calculate_io_vector_overlaps.R contains the functions to calculate IOC and OOC similarities.
- sample_synthetic_firm_level_shocks.R contains the functions to sample synthetic shocks that are heterogenous on the firm-level, but homogenous on the sector-level.
