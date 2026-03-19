# evaluate.py (Final Fix for G-IoU)
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import config
from dataset import RadarFileDataset
from metrics import TrackingMetrics
from model import GNNGroupTracker

# --- Import Trackers ---
from trackers.baseline import BaselineTracker
from trackers.cbmember import CBMeMBerTracker
from trackers.gm_cphd import GMCPHDTracker
from trackers.gnn_processor import GNNPostProcessor
from trackers.graph_mb import GraphMBTracker


def unpack_group_offsets(model_out):
    if hasattr(model_out, 'get_offsets'):
        return model_out.get_offsets('group')
    if isinstance(model_out, (tuple, list)) and len(model_out) >= 2:
        return model_out[1]
    raise ValueError('模型输出缺少 group offset。')


def run_evaluation():
    device = torch.device(config.DEVICE)
    print(f"Loading Test Data from {config.DATA_ROOT}...")
    test_set = RadarFileDataset('test')
    if len(test_set) == 0:
        return

    # Load GNN
    gnn_model = GNNGroupTracker(
        input_node_dim=config.INPUT_DIM,
        input_edge_dim=config.EDGE_DIM,
        hidden_dim=config.HIDDEN_DIM,
    ).to(device)
    if os.path.exists(config.MODEL_USE_PATH):
        gnn_model.load_state_dict(torch.load(config.MODEL_USE_PATH, map_location=device))
        gnn_model.eval()
    else:
        gnn_model = None

    # Instantiate Trackers
    baseline_tracker = BaselineTracker(eps=35, min_samples=3)
    gm_cphd_tracker = GMCPHDTracker()
    cbmember_tracker = CBMeMBerTracker()
    graph_mb_tracker = GraphMBTracker()

    metrics = {
        'Baseline (DBSCAN+KF)': TrackingMetrics(),
        'GM-CPHD (Standard)': TrackingMetrics(),
        'CBMeMBer (Standard)': TrackingMetrics(),
        'Graph-MB (Paper)': TrackingMetrics(),
        'H-GAT-GT (Ours)': TrackingMetrics(),
    }

    print('Running Evaluation Loop (5 Trackers)...')
    for episode_idx in tqdm(range(len(test_set))):
        episode_graphs = test_set.get(episode_idx)

        # Reset Trackers
        gnn_processor = GNNPostProcessor()
        baseline_tracker.reset()
        gm_cphd_tracker.reset()
        cbmember_tracker.reset()
        graph_mb_tracker.reset()

        for graph in episode_graphs:
            gt_data = graph.gt_centers.numpy()
            gt_centers = gt_data[:, 1:3] if len(gt_data) > 0 else np.zeros((0, 2))
            gt_ids = gt_data[:, 0].astype(int) if len(gt_data) > 0 else []
            meas_points = graph.x.numpy()

            # --- 1. H-GAT-GT (Ours) ---
            t0 = time.time()
            pred_c_gnn, pred_id_gnn = np.array([]), np.array([])
            point_to_track_map_gnn = np.full(len(meas_points), -1)
            pred_shapes_gnn = None

            if gnn_model:
                graph_dev = graph.to(device)
                with torch.no_grad():
                    model_out = gnn_model(graph_dev)
                    offsets = unpack_group_offsets(model_out)

                corrected_pos = meas_points + offsets.detach().cpu().numpy()

                labels = np.array([])
                if len(corrected_pos) > 0:
                    try:
                        labels = DBSCAN(eps=30, min_samples=3).fit(corrected_pos).labels_
                    except Exception:
                        pass

                det_centers, det_shapes = [], []
                cluster_indices_list = []

                if len(labels) > 0:
                    for label in set(labels):
                        if label == -1:
                            continue
                        indices = np.where(labels == label)[0]
                        cluster_indices_list.append(indices)

                        det_centers.append(np.mean(corrected_pos[indices], axis=0))

                        pts_raw = meas_points[indices]
                        if len(pts_raw) > 1:
                            lower = np.percentile(pts_raw, 5, axis=0)
                            upper = np.percentile(pts_raw, 95, axis=0)
                            wh = upper - lower
                        else:
                            wh = np.array([0.0, 0.0])
                        det_shapes.append(np.maximum(wh, 3.0))

                det_centers = np.array(det_centers).reshape(-1, 2)
                det_shapes = np.array(det_shapes).reshape(-1, 2)
                if len(det_shapes) == 0:
                    det_shapes = None

                if det_centers.shape[0] > 0:
                    pred_c_gnn, pred_id_gnn, pred_shapes_gnn = gnn_processor.update(det_centers, det_shapes)
                else:
                    pred_c_gnn, pred_id_gnn, pred_shapes_gnn = gnn_processor.update(np.empty((0, 2)), None)

                if len(pred_c_gnn) > 0 and len(det_centers) > 0:
                    cost_matrix = euclidean_distances(pred_c_gnn, det_centers)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for r, c in zip(row_ind, col_ind):
                        if cost_matrix[r, c] < 20.0:
                            track_id = pred_id_gnn[r]
                            point_indices = cluster_indices_list[c]
                            point_to_track_map_gnn[point_indices] = track_id

            t1 = time.time()
            metrics['H-GAT-GT (Ours)'].update_time(t1 - t0)

            gt_shapes_list = []
            pt_lbl = graph.point_labels.cpu().numpy()
            for gid in gt_ids:
                idx = np.where(pt_lbl == gid)[0]
                if len(idx) > 1:
                    pts = meas_points[idx]
                    lower = np.percentile(pts, 5, axis=0)
                    upper = np.percentile(pts, 95, axis=0)
                    gt_shapes_list.append(np.maximum(upper - lower, 3.0))
                else:
                    gt_shapes_list.append(np.array([3.0, 3.0]))
            gt_shapes_arr = np.array(gt_shapes_list).reshape(-1, 2) if len(gt_shapes_list) > 0 else None

            metrics['H-GAT-GT (Ours)'].update(
                gt_centers,
                gt_ids,
                pred_c_gnn,
                pred_id_gnn,
                gt_shapes=gt_shapes_arr,
                pred_shapes=pred_shapes_gnn,
            )
            metrics['H-GAT-GT (Ours)'].update_clustering_metrics(graph.point_labels.cpu().numpy(), point_to_track_map_gnn)

            # --- Pre-process for Baselines ---
            t_pre = time.time()
            if len(meas_points) > 0:
                db_labels = DBSCAN(eps=35, min_samples=3).fit_predict(meas_points)
                detected_centroids = []
                centroid_to_points = {}
                valid_l = [label for label in set(db_labels) if label != -1]
                for i, label in enumerate(valid_l):
                    idx = np.where(db_labels == label)[0]
                    detected_centroids.append(np.mean(meas_points[idx], axis=0))
                    centroid_to_points[i] = idx
            else:
                detected_centroids = []
                centroid_to_points = {}
            t_pre_end = time.time()
            pre_time = t_pre_end - t_pre

            def get_rfs_point_labels(rfs_c, rfs_id):
                pt_lbl_inner = np.full(len(meas_points), -1)
                if len(rfs_c) > 0 and len(detected_centroids) > 0:
                    cost = euclidean_distances(rfs_c, detected_centroids)
                    r, c = linear_sum_assignment(cost)
                    for r_i, c_i in zip(r, c):
                        if cost[r_i, c_i] < 20.0:
                            tid = rfs_id[r_i]
                            if c_i in centroid_to_points:
                                pt_lbl_inner[centroid_to_points[c_i]] = tid
                return pt_lbl_inner

            # --- 2. Baseline ---
            t0 = time.time()
            base_c, base_id, base_pt_lbl = baseline_tracker.step(meas_points)
            t1 = time.time()
            metrics['Baseline (DBSCAN+KF)'].update_time(t1 - t0)
            metrics['Baseline (DBSCAN+KF)'].update(gt_centers, gt_ids, base_c, base_id)
            metrics['Baseline (DBSCAN+KF)'].update_clustering_metrics(graph.point_labels.cpu().numpy(), base_pt_lbl)

            # --- 3. GM-CPHD ---
            t0 = time.time()
            scphd_c, scphd_id = gm_cphd_tracker.step(detected_centroids)
            t1 = time.time()
            metrics['GM-CPHD (Standard)'].update_time(t1 - t0 + pre_time)
            metrics['GM-CPHD (Standard)'].update(gt_centers, gt_ids, scphd_c, scphd_id)
            metrics['GM-CPHD (Standard)'].update_clustering_metrics(
                graph.point_labels.cpu().numpy(), get_rfs_point_labels(scphd_c, scphd_id)
            )

            # --- 4. CBMeMBer ---
            t0 = time.time()
            cb_c, cb_id = cbmember_tracker.step(detected_centroids)
            t1 = time.time()
            metrics['CBMeMBer (Standard)'].update_time(t1 - t0 + pre_time)
            metrics['CBMeMBer (Standard)'].update(gt_centers, gt_ids, cb_c, cb_id)
            metrics['CBMeMBer (Standard)'].update_clustering_metrics(
                graph.point_labels.cpu().numpy(), get_rfs_point_labels(cb_c, cb_id)
            )

            # --- 5. Graph-MB (Paper) ---
            t0 = time.time()
            gmb_c, gmb_id, gmb_pt_lbl = graph_mb_tracker.step(meas_points)
            t1 = time.time()
            metrics['Graph-MB (Paper)'].update_time(t1 - t0)
            metrics['Graph-MB (Paper)'].update(gt_centers, gt_ids, gmb_c, gmb_id)
            metrics['Graph-MB (Paper)'].update_clustering_metrics(graph.point_labels.cpu().numpy(), gmb_pt_lbl)

    final_res = {k: v.compute() for k, v in metrics.items()}
    df = pd.DataFrame(final_res).T
    cols = [
        'MOTA', 'MOTP', 'IDSW', 'FAR', 'OSPA (Total)', 'OSPA (Loc)', 'OSPA (Card)',
        'RMSE (Pos)', 'Count Err', 'Purity', 'Completeness', 'G-IoU', 'Time', 'Comp',
    ]
    df = df[[c for c in cols if c in df.columns]]

    print("\n" + "=" * 100)
    print("FINAL 5-WAY COMPARISON RESULTS")
    print("=" * 100)
    print(df.to_string())
    print("=" * 100)

    plot_cats = ['MOTA', 'OSPA (Total)', 'RMSE (Pos)', 'Count Err', 'Time']
    x = np.arange(len(plot_cats))
    width = 0.15
    fig, ax = plt.subplots(figsize=(18, 8))
    trackers = [
        'Baseline (DBSCAN+KF)',
        'GM-CPHD (Standard)',
        'CBMeMBer (Standard)',
        'Graph-MB (Paper)',
        'H-GAT-GT (Ours)',
    ]

    for i, trk_name in enumerate(trackers):
        vals = [final_res[trk_name].get(c, 0) for c in plot_cats]
        viz_vals = []
        for value, cat in zip(vals, plot_cats):
            if cat in ['OSPA (Total)', 'RMSE (Pos)', 'Count Err', 'Time']:
                viz_vals.append(-value)
            else:
                viz_vals.append(value)

        rects = ax.bar(x + (i - 2) * width, viz_vals, width, label=trk_name)
        for rect, value in zip(rects, vals):
            height = rect.get_height()
            ax.annotate(
                f'{value:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -15),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8,
            )

    ax.set_ylabel('Score (Higher is Better / Negative is Cost)')
    ax.set_title('Comprehensive 5-Way Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_cats)
    ax.legend()
    plt.tight_layout()
    plt.savefig('final_benchmark_5way.png')


if __name__ == '__main__':
    run_evaluation()
