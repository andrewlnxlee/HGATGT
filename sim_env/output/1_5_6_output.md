====================================================================================================
FINAL POINT-LEVEL RESULTS (DETECTED-ONLY GT)
====================================================================================================
Note: point-level MOTA is auxiliary under detected-only GT; interpret OSPA / RMSE / IDSW first.
Prediction-side point filtering: DBSCAN eps=35.0, min_samples=3; only non-noise points enter point tracking.
Point tracker thresholds: stage1=20.0, recovery=28.0, max_age=15, metric_match=28.0.
Point IDSW gap tolerance: 1 frame(s).
H-GAT-GT point IDs now use group tracking plus group-constrained point association.
Diagnostic row included: H-GAT-GT (Meas Ablation) tracks filtered raw meas_points to compare against corrected_pos.
                          OSPA (Total)  OSPA (Loc)  OSPA (Card)  RMSE (Pos)     IDSW  Count Err      MOTA      MOTP     FAR       Time
Baseline (DBSCAN+KF)          4.482426    1.949287     2.957385    2.138924  39098.0     2.3052  0.688386  1.880404  0.1912   0.761093
GM-CPHD (Standard)            2.814555    2.015440     1.110236    2.119639  48037.0     0.2448  0.697798  1.876149  0.2054   0.772956
CBMeMBer (Standard)           4.885194    1.891580     3.387062    2.135331  41955.0     2.8246  0.655436  1.879955  0.1828   0.390926
Graph-MB (Paper)              4.424994    1.995006     2.975242    2.155769  46985.0     1.5186  0.663419  1.884417  0.2404  13.691570
H-GAT-GT (Ours)               2.509186    1.667112     1.111603    1.755109  44064.0     0.2452  0.722159  1.532153  0.2056   3.957639
H-GAT-GT (Meas Ablation)      2.814555    2.015440     1.110236    2.119639  46194.0     0.2448  0.709098  1.876149  0.2054   1.273987
====================================================================================================