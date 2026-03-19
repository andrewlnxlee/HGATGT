====================================================================================================
FINAL POINT-LEVEL RESULTS (DETECTED-ONLY GT)
====================================================================================================
Note: point-level MOTA is auxiliary under detected-only GT; interpret OSPA / RMSE / IDSW first.
Prediction-side point filtering: DBSCAN eps=35.0, min_samples=3; only non-noise points enter point tracking.
Point tracker thresholds: stage1=14.0, recovery=22.0, max_age=20, metric_match=28.0.
Diagnostic row included: H-GAT-GT (Meas Ablation) tracks filtered raw meas_points to compare against corrected_pos.
                          OSPA (Total)  OSPA (Loc)  OSPA (Card)  RMSE (Pos)      IDSW  Count Err      MOTA      MOTP     FAR       Time
Baseline (DBSCAN+KF)          4.482426    1.949287     2.957385    2.138924   42426.0     2.3052  0.667981  1.880404  0.1912   0.759137
GM-CPHD (Standard)            2.814555    2.015440     1.110236    2.119639   48343.0     0.2448  0.695922  1.876149  0.2054   0.774931
CBMeMBer (Standard)           4.885194    1.891580     3.387062    2.135331   44797.0     2.8246  0.638010  1.879955  0.1828   0.393455
Graph-MB (Paper)              4.424994    1.995006     2.975242    2.155769   48291.0     1.5186  0.655411  1.884417  0.2404  13.807373
H-GAT-GT (Ours)               2.509187    1.667112     1.111603    1.755110  125473.0     0.2452  0.223002  1.532154  0.2056   4.365033
H-GAT-GT (Meas Ablation)      2.814555    2.015440     1.110236    2.119639  125959.0     0.2448  0.220022  1.876149  0.2054   1.702894
====================================================================================================