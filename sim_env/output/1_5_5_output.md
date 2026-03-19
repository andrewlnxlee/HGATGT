====================================================================================================
FINAL POINT-LEVEL RESULTS (DETECTED-ONLY GT)
====================================================================================================
Note: point-level MOTA is auxiliary under detected-only GT; interpret OSPA / RMSE / IDSW first.
Prediction-side point filtering: DBSCAN eps=35.0, min_samples=3; only non-noise points enter point tracking.
Point tracker thresholds: stage1=20.0, recovery=28.0, max_age=15, metric_match=28.0.
Point IDSW gap tolerance: 0 frame(s).
Diagnostic row included: H-GAT-GT (Meas Ablation) tracks filtered raw meas_points to compare against corrected_pos.
                          OSPA (Total)  OSPA (Loc)  OSPA (Card)  RMSE (Pos)      IDSW  Count Err      MOTA      MOTP     FAR       Time
Baseline (DBSCAN+KF)          4.482426    1.949287     2.957385    2.138924   31835.0     2.3052  0.732919  1.880404  0.1912   0.757420
GM-CPHD (Standard)            2.814555    2.015440     1.110236    2.119639   42349.0     0.2448  0.732674  1.876149  0.2054   0.784790
CBMeMBer (Standard)           4.885194    1.891580     3.387062    2.135331   32238.0     2.8246  0.715015  1.879955  0.1828   0.397286
Graph-MB (Paper)              4.424994    1.995006     2.975242    2.155769   40262.0     1.5186  0.704641  1.884417  0.2404  13.991363
H-GAT-GT (Ours)               2.509187    1.667112     1.111603    1.755109  103764.0     0.2452  0.356110  1.532154  0.2056   3.970065
H-GAT-GT (Meas Ablation)      2.814555    2.015440     1.110236    2.119639  105435.0     0.2448  0.345864  1.876149  0.2054   1.308680
====================================================================================================
root@0c8156b0cff2:/workspace# python sim_env/evaluate_single.py
Loading Test Data from ./data...
Running Point-Level Evaluation Loop (detected-only GT)...
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:52<00:00,  1.12s/it]

====================================================================================================
FINAL POINT-LEVEL RESULTS (DETECTED-ONLY GT)
====================================================================================================
Note: point-level MOTA is auxiliary under detected-only GT; interpret OSPA / RMSE / IDSW first.
Prediction-side point filtering: DBSCAN eps=35.0, min_samples=3; only non-noise points enter point tracking.
Point tracker thresholds: stage1=20.0, recovery=28.0, max_age=15, metric_match=28.0.
Point IDSW gap tolerance: 1 frame(s).
Diagnostic row included: H-GAT-GT (Meas Ablation) tracks filtered raw meas_points to compare against corrected_pos.
                          OSPA (Total)  OSPA (Loc)  OSPA (Card)  RMSE (Pos)      IDSW  Count Err      MOTA      MOTP     FAR       Time
Baseline (DBSCAN+KF)          4.482426    1.949287     2.957385    2.138924   39098.0     2.3052  0.688386  1.880404  0.1912   0.752133
GM-CPHD (Standard)            2.814555    2.015440     1.110236    2.119639   48037.0     0.2448  0.697798  1.876149  0.2054   0.777900
CBMeMBer (Standard)           4.885194    1.891580     3.387062    2.135331   41955.0     2.8246  0.655436  1.879955  0.1828   0.392686
Graph-MB (Paper)              4.424994    1.995006     2.975242    2.155769   46985.0     1.5186  0.663419  1.884417  0.2404  13.888472
H-GAT-GT (Ours)               2.509187    1.667112     1.111603    1.755109  110596.0     0.2452  0.314219  1.532154  0.2056   3.921697
H-GAT-GT (Meas Ablation)      2.814555    2.015440     1.110236    2.119639  112302.0     0.2448  0.303759  1.876149  0.2054   1.297160
====================================================================================================