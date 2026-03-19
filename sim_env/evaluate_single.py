import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from trackers.baseline import BaselineTracker
from trackers.cbmember import CBMeMBerTracker
from trackers.gm_cphd import GMCPHDTracker
from trackers.gnn_processor_single import GNNPointPostProcessor
from trackers.graph_mb import GraphMBTracker


POINT_MATCH_THRESHOLD = 15.0
POINT_OSPA_C = 25.0
GROUP_TO_POINT_ASSOC_THRESH = 45.0
GROUP_TO_CENTROID_THRESH = 20.0


class GroupConstrainedPointAssociator:
    def __init__(self, match_threshold=GROUP_TO_POINT_ASSOC_THRESH, max_age=2):
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, points, group_ids):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        group_ids = np.asarray(group_ids, dtype=int).reshape(-1)

        for trk in self.tracks.values():
            trk['pred_pos'] = trk['pos'] + trk['vel']
            trk['age'] += 1

        valid_group_ids = sorted(int(gid) for gid in np.unique(group_ids) if gid >= 0)
        for gid in valid_group_ids:
            det_indices = np.where(group_ids == gid)[0]
            if len(det_indices) == 0:
                continue

            track_ids = [tid for tid, trk in self.tracks.items() if trk['group_id'] == gid]
            matched_det_indices = set()

            if track_ids:
                pred_pos = np.array([self.tracks[tid]['pred_pos'] for tid in track_ids])
                det_points = points[det_indices]
                cost = euclidean_distances(pred_pos, det_points)
                row_ind, col_ind = linear_sum_assignment(cost)

                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] > self.match_threshold:
                        continue
                    tid = track_ids[r]
                    det_idx = det_indices[c]
                    trk = self.tracks[tid]
                    new_pos = points[det_idx]
                    displacement = new_pos - trk['pos']
                    trk['vel'] = 0.5 * trk['vel'] + 0.5 * displacement
                    trk['pos'] = new_pos
                    trk['pred_pos'] = new_pos
                    trk['group_id'] = gid
                    trk['age'] = 0
                    matched_det_indices.add(det_idx)

            for det_idx in det_indices:
                if det_idx in matched_det_indices:
                    continue
                self.tracks[self.next_id] = {
                    'pos': points[det_idx].copy(),
                    'pred_pos': points[det_idx].copy(),
                    'vel': np.zeros(2, dtype=float),
                    'group_id': gid,
                    'age': 0,
                }
                self.next_id += 1

        stale_ids = [tid for tid, trk in self.tracks.items() if trk['age'] > self.max_age]
        for tid in stale_ids:
            del self.tracks[tid]

        active_ids = [tid for tid, trk in self.tracks.items() if trk['age'] == 0]
        active_ids.sort()
        if not active_ids:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)

        pred_points = np.array([self.tracks[tid]['pos'] for tid in active_ids], dtype=float)
        pred_ids = np.array(active_ids, dtype=int)
        return pred_points, pred_ids


def ensure_point_gt(graph, episode_idx, frame_idx):
    if not getattr(graph, 'has_gt_points', False) or not getattr(graph, 'has_point_ids', False):
        raise ValueError(
            f"sample {episode_idx} frame {frame_idx} 缺少 gt_points/point_ids；请先用更新后的 sim_env/generate_data.py 重生成数据。"
        )


def cluster_measurements(points, eps=35, min_samples=3):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, 2), dtype=float), {}

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    centroids = []
    centroid_to_points = {}
    valid_labels = sorted(l for l in np.unique(labels) if l != -1)
    for idx, label in enumerate(valid_labels):
        point_indices = np.where(labels == label)[0]
        centroids.append(np.mean(points[point_indices], axis=0))
        centroid_to_points[idx] = point_indices

    centroids = np.asarray(centroids, dtype=float).reshape(-1, 2)
    return labels, centroids, centroid_to_points


def project_cluster_tracks_to_points(points, cluster_labels, track_centers, track_ids, dist_thresh=GROUP_TO_CENTROID_THRESH):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    cluster_labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    point_group_ids = np.full(len(points), -1, dtype=int)

    valid_labels = sorted(l for l in np.unique(cluster_labels) if l != -1)
    if len(valid_labels) == 0:
        return point_group_ids

    cluster_centers = []
    cluster_to_points = {}
    for idx, label in enumerate(valid_labels):
        indices = np.where(cluster_labels == label)[0]
        cluster_centers.append(np.mean(points[indices], axis=0))
        cluster_to_points[idx] = indices

    cluster_centers = np.asarray(cluster_centers, dtype=float).reshape(-1, 2)
    track_centers = np.asarray(track_centers, dtype=float).reshape(-1, 2)
    track_ids = np.asarray(track_ids, dtype=int).reshape(-1)

    if len(cluster_centers) == 0 or len(track_centers) == 0:
        return point_group_ids

    cost = euclidean_distances(track_centers, cluster_centers)
    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > dist_thresh:
            continue
        point_group_ids[cluster_to_points[c]] = track_ids[r]
    return point_group_ids


def get_rfs_point_group_ids(points, detected_centroids, centroid_to_points, rfs_centers, rfs_ids, dist_thresh=GROUP_TO_CENTROID_THRESH):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    point_group_ids = np.full(len(points), -1, dtype=int)
    detected_centroids = np.asarray(detected_centroids, dtype=float).reshape(-1, 2)
    rfs_centers = np.asarray(rfs_centers, dtype=float).reshape(-1, 2)
    rfs_ids = np.asarray(rfs_ids, dtype=int).reshape(-1)

    if len(detected_centroids) == 0 or len(rfs_centers) == 0:
        return point_group_ids

    cost = euclidean_distances(rfs_centers, detected_centroids)
    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > dist_thresh:
            continue
        if c in centroid_to_points:
            point_group_ids[centroid_to_points[c]] = rfs_ids[r]
    return point_group_ids


def create_metrics():
    return TrackingMetrics(ospa_c=POINT_OSPA_C, match_threshold=POINT_MATCH_THRESHOLD)


def run_evaluation():
    device = torch.device(config.DEVICE)
    print(f"Loading Test Data from {config.DATA_ROOT}...")
    test_set = RadarFileDataset('test', include_empty=True)
    if len(test_set) == 0:
        return

    gnn_model = GNNGroupTracker().to(device)
    if os.path.exists(config.MODEL_USE_PATH):
        gnn_model.load_state_dict(torch.load(config.MODEL_USE_PATH, map_location=device))
        gnn_model.eval()
    else:
        gnn_model = None

    baseline_tracker = BaselineTracker(eps=35, min_samples=3)
    gm_cphd_tracker = GMCPHDTracker()
    cbmember_tracker = CBMeMBerTracker()
    graph_mb_tracker = GraphMBTracker()

    metrics = {
        'Baseline (DBSCAN+KF)': create_metrics(),
        'GM-CPHD (Standard)': create_metrics(),
        'CBMeMBer (Standard)': create_metrics(),
        'Graph-MB (Paper)': create_metrics(),
        'H-GAT-GT (Ours)': create_metrics(),
    }

    print('Running Point-Level Evaluation Loop (5 Trackers)...')
    for episode_idx in tqdm(range(len(test_set))):
        episode_graphs = test_set.get(episode_idx)

        gnn_processor = GNNPointPostProcessor()
        baseline_tracker.reset()
        gm_cphd_tracker.reset()
        cbmember_tracker.reset()
        graph_mb_tracker.reset()

        baseline_point_assoc = GroupConstrainedPointAssociator()
        gm_point_assoc = GroupConstrainedPointAssociator()
        cb_point_assoc = GroupConstrainedPointAssociator()
        gmb_point_assoc = GroupConstrainedPointAssociator()

        for metric in metrics.values():
            metric.reset_sequence()

        for frame_idx, graph in enumerate(episode_graphs):
            ensure_point_gt(graph, episode_idx, frame_idx)

            gt_points_data = graph.gt_points.cpu().numpy()
            gt_pos = gt_points_data[:, 1:3] if len(gt_points_data) > 0 else np.zeros((0, 2), dtype=float)
            gt_ids = gt_points_data[:, 0].astype(int) if len(gt_points_data) > 0 else np.zeros((0,), dtype=int)
            meas_points = graph.x.cpu().numpy()

            _, detected_centroids, centroid_to_points = cluster_measurements(meas_points, eps=35, min_samples=3)

            # H-GAT-GT
            t0 = time.time()
            pred_pos_gnn = np.zeros((0, 2), dtype=float)
            pred_id_gnn = np.zeros((0,), dtype=int)
            if gnn_model is not None and len(meas_points) > 0:
                if graph.edge_index.shape[1] > 0:
                    graph_dev = graph.to(device)
                    with torch.no_grad():
                        out = gnn_model(graph_dev)
                        if isinstance(out, tuple):
                            offsets = out[1]
                        else:
                            offsets = out[1]
                    corrected_pos = meas_points + offsets.detach().cpu().numpy()
                else:
                    corrected_pos = meas_points
                pred_pos_gnn, pred_id_gnn = gnn_processor.update(corrected_pos)
            else:
                pred_pos_gnn, pred_id_gnn = gnn_processor.update(np.empty((0, 2)))
            t1 = time.time()
            metrics['H-GAT-GT (Ours)'].update_time(t1 - t0)
            metrics['H-GAT-GT (Ours)'].update(gt_pos, gt_ids, pred_pos_gnn, pred_id_gnn)

            # Baseline
            t0 = time.time()
            base_c, base_id, base_cluster_labels = baseline_tracker.step(meas_points)
            base_group_ids = project_cluster_tracks_to_points(meas_points, base_cluster_labels, base_c, base_id)
            base_pred_pos, base_pred_id = baseline_point_assoc.update(meas_points, base_group_ids)
            t1 = time.time()
            metrics['Baseline (DBSCAN+KF)'].update_time(t1 - t0)
            metrics['Baseline (DBSCAN+KF)'].update(gt_pos, gt_ids, base_pred_pos, base_pred_id)

            # GM-CPHD
            t0 = time.time()
            scphd_c, scphd_id = gm_cphd_tracker.step(detected_centroids)
            scphd_group_ids = get_rfs_point_group_ids(meas_points, detected_centroids, centroid_to_points, scphd_c, scphd_id)
            scphd_pred_pos, scphd_pred_id = gm_point_assoc.update(meas_points, scphd_group_ids)
            t1 = time.time()
            metrics['GM-CPHD (Standard)'].update_time(t1 - t0)
            metrics['GM-CPHD (Standard)'].update(gt_pos, gt_ids, scphd_pred_pos, scphd_pred_id)

            # CBMeMBer
            t0 = time.time()
            cb_c, cb_id = cbmember_tracker.step(detected_centroids)
            cb_group_ids = get_rfs_point_group_ids(meas_points, detected_centroids, centroid_to_points, cb_c, cb_id)
            cb_pred_pos, cb_pred_id = cb_point_assoc.update(meas_points, cb_group_ids)
            t1 = time.time()
            metrics['CBMeMBer (Standard)'].update_time(t1 - t0)
            metrics['CBMeMBer (Standard)'].update(gt_pos, gt_ids, cb_pred_pos, cb_pred_id)

            # Graph-MB
            t0 = time.time()
            _, _, gmb_group_ids = graph_mb_tracker.step(meas_points)
            gmb_pred_pos, gmb_pred_id = gmb_point_assoc.update(meas_points, gmb_group_ids)
            t1 = time.time()
            metrics['Graph-MB (Paper)'].update_time(t1 - t0)
            metrics['Graph-MB (Paper)'].update(gt_pos, gt_ids, gmb_pred_pos, gmb_pred_id)

    final_res = {name: metric.compute() for name, metric in metrics.items()}
    df = pd.DataFrame(final_res).T
    cols = [
        'MOTA', 'MOTP', 'IDSW', 'FAR', 'OSPA (Total)', 'OSPA (Loc)', 'OSPA (Card)',
        'RMSE (Pos)', 'Count Err', 'Time'
    ]
    df = df[[c for c in cols if c in df.columns]]

    print('\n' + '=' * 100)
    print('FINAL 5-WAY POINT-LEVEL RESULTS')
    print('=' * 100)
    print(df.to_string())
    print('=' * 100)


if __name__ == '__main__':
    run_evaluation()
