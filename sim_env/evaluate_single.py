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
from trackers.gnn_processor import GNNPostProcessor
from trackers.gnn_processor_single import (
    POINT_TRACK_MAX_AGE as DEFAULT_POINT_TRACK_MAX_AGE,
    POINT_TRACK_RECOVERY_THRESHOLD as DEFAULT_POINT_TRACK_RECOVERY_THRESHOLD,
    POINT_TRACK_STAGE1_THRESHOLD as DEFAULT_POINT_TRACK_STAGE1_THRESHOLD,
)
from trackers.graph_mb import GraphMBTracker


POINT_TRACK_STAGE1_THRESHOLD = float(
    getattr(config, 'POINT_TRACK_STAGE1_THRESHOLD', DEFAULT_POINT_TRACK_STAGE1_THRESHOLD)
)
POINT_TRACK_RECOVERY_THRESHOLD = float(
    getattr(config, 'POINT_TRACK_RECOVERY_THRESHOLD', DEFAULT_POINT_TRACK_RECOVERY_THRESHOLD)
)
POINT_TRACK_MAX_AGE = int(getattr(config, 'POINT_TRACK_MAX_AGE', DEFAULT_POINT_TRACK_MAX_AGE))
POINT_MATCH_THRESHOLD = float(getattr(config, 'POINT_MATCH_THRESHOLD', POINT_TRACK_RECOVERY_THRESHOLD))
POINT_ID_SWITCH_GAP_TOLERANCE = getattr(config, 'POINT_ID_SWITCH_GAP_TOLERANCE', 0)
POINT_CLUSTER_EPS = float(getattr(config, 'POINT_CLUSTER_EPS', 35))
POINT_CLUSTER_MIN_SAMPLES = int(getattr(config, 'POINT_CLUSTER_MIN_SAMPLES', 3))
POINT_OSPA_C = 25.0
GROUP_TO_POINT_ASSOC_THRESH = float(getattr(config, 'GROUP_TO_POINT_ASSOC_THRESH', POINT_TRACK_RECOVERY_THRESHOLD))
GROUP_TO_CENTROID_THRESH = float(getattr(config, 'GROUP_TO_CENTROID_THRESH', POINT_TRACK_STAGE1_THRESHOLD))
GROUP_POINT_MAX_AGE = int(getattr(config, 'GROUP_POINT_MAX_AGE', min(POINT_TRACK_MAX_AGE, 4)))
ENABLE_MEAS_DIAGNOSTIC = bool(getattr(config, 'ENABLE_MEAS_DIAGNOSTIC', True))
ENABLE_POINT_UNCERTAINTY_GATING = bool(getattr(config, 'ENABLE_POINT_UNCERTAINTY_GATING', False))
ENABLE_POINT_UNCERTAINTY_ABLATION = bool(getattr(config, 'ENABLE_POINT_UNCERTAINTY_ABLATION', False))


class GroupConstrainedPointAssociator:
    def __init__(self, match_threshold=GROUP_TO_POINT_ASSOC_THRESH, max_age=GROUP_POINT_MAX_AGE):
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
                    if cost[r, c] >= self.match_threshold:
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


def extract_detected_point_gt(graph, episode_idx, frame_idx):
    gt_points_data = graph.gt_points.cpu().numpy()
    point_ids = graph.point_ids.cpu().numpy().astype(int).reshape(-1)
    detected_ids = point_ids[point_ids > 0]
    if len(detected_ids) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)

    _, first_idx = np.unique(detected_ids, return_index=True)
    detected_ids = detected_ids[np.sort(first_idx)]

    gt_lookup = {int(row[0]): np.asarray(row[1:3], dtype=float) for row in gt_points_data}
    missing_ids = [point_id for point_id in detected_ids if point_id not in gt_lookup]
    if missing_ids:
        raise ValueError(
            f"sample {episode_idx} frame {frame_idx} 的 detected-only GT 缺少 member_id={missing_ids[:5]} 的真值坐标。"
        )

    gt_pos = np.array([gt_lookup[point_id] for point_id in detected_ids], dtype=float).reshape(-1, 2)
    gt_ids = detected_ids.astype(int)
    return gt_pos, gt_ids


def unpack_head_outputs(model_out, head='group'):
    if hasattr(model_out, 'get_offsets') and hasattr(model_out, 'get_uncertainty'):
        return model_out.get_offsets(head), model_out.get_uncertainty(head)

    if not isinstance(model_out, (tuple, list)):
        raise ValueError('模型输出格式不受支持。')

    if len(model_out) >= 5:
        if head == 'group':
            return model_out[1], model_out[2]
        if head == 'point':
            return model_out[3], model_out[4]
    if len(model_out) >= 3:
        return model_out[1], model_out[2]
    if len(model_out) >= 2:
        return model_out[1], None
    raise ValueError('模型输出缺少 offset 信息。')


def apply_point_uncertainty_gating(offsets, uncertainty, enabled=ENABLE_POINT_UNCERTAINTY_GATING):
    if uncertainty is None or not enabled:
        return offsets

    gate_scale = float(getattr(config, 'POINT_UNCERTAINTY_GATE_SCALE', 1.0))
    uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1, 2)
    mean_uncertainty = np.mean(uncertainty, axis=1, keepdims=True)
    gains = 1.0 / (1.0 + gate_scale * mean_uncertainty)
    return offsets * gains


def infer_corrected_points(gnn_model, graph, meas_points, device, head='point', use_uncertainty_gating=None):
    meas_points = np.asarray(meas_points, dtype=float).reshape(-1, 2)
    if gnn_model is None or len(meas_points) == 0 or graph.edge_index.shape[1] == 0:
        return meas_points

    graph_dev = graph.to(device)
    with torch.no_grad():
        model_out = gnn_model(graph_dev)
        offsets, uncertainty = unpack_head_outputs(model_out, head=head)

    offsets = offsets.detach().cpu().numpy().reshape(-1, 2)
    if len(offsets) != len(meas_points):
        raise ValueError(f"offset 数量与 meas 数量不一致: {len(offsets)} vs {len(meas_points)}")

    if uncertainty is not None and head == 'point':
        uncertainty = uncertainty.detach().cpu().numpy().reshape(-1, 2)
        if use_uncertainty_gating is None:
            use_uncertainty_gating = ENABLE_POINT_UNCERTAINTY_GATING
        offsets = apply_point_uncertainty_gating(offsets, uncertainty, enabled=use_uncertainty_gating)

    return meas_points + offsets


def cluster_measurements(points, eps=POINT_CLUSTER_EPS, min_samples=POINT_CLUSTER_MIN_SAMPLES):
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
        if cost[r, c] >= dist_thresh:
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
        if cost[r, c] >= dist_thresh:
            continue
        if c in centroid_to_points:
            point_group_ids[centroid_to_points[c]] = rfs_ids[r]
    return point_group_ids


def create_metrics():
    return TrackingMetrics(
        ospa_c=POINT_OSPA_C,
        match_threshold=POINT_MATCH_THRESHOLD,
        id_switch_gap_tolerance=POINT_ID_SWITCH_GAP_TOLERANCE,
    )


def filter_clustered_points(points, eps=POINT_CLUSTER_EPS, min_samples=POINT_CLUSTER_MIN_SAMPLES):
    labels, _, _ = cluster_measurements(points, eps=eps, min_samples=min_samples)
    keep_mask = labels != -1
    filtered_points = np.asarray(points, dtype=float).reshape(-1, 2)[keep_mask]
    return filtered_points, keep_mask, labels


def build_group_detections(points, eps=POINT_CLUSTER_EPS, min_samples=POINT_CLUSTER_MIN_SAMPLES):
    cluster_labels, det_centers, centroid_to_points = cluster_measurements(points, eps=eps, min_samples=min_samples)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    det_shapes = []
    valid_labels = sorted(l for l in np.unique(cluster_labels) if l != -1)
    for label in valid_labels:
        indices = np.where(cluster_labels == label)[0]
        pts = points[indices]
        if len(pts) > 1:
            lower = np.percentile(pts, 5, axis=0)
            upper = np.percentile(pts, 95, axis=0)
            wh = upper - lower
        else:
            wh = np.array([0.0, 0.0])
        det_shapes.append(np.maximum(wh, 3.0))

    det_shapes = np.asarray(det_shapes, dtype=float).reshape(-1, 2)
    if len(det_shapes) == 0:
        det_shapes = None
    return cluster_labels, det_centers, centroid_to_points, det_shapes


def run_hgat_point_identity_pipeline(points, group_tracker, point_assoc):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    cluster_labels, det_centers, _, det_shapes = build_group_detections(points)
    if len(det_centers) > 0:
        pred_group_centers, pred_group_ids, _ = group_tracker.update(det_centers, det_shapes)
    else:
        pred_group_centers, pred_group_ids, _ = group_tracker.update(np.empty((0, 2)), None)

    point_group_ids = project_cluster_tracks_to_points(points, cluster_labels, pred_group_centers, pred_group_ids)
    pred_points, pred_ids = point_assoc.update(points, point_group_ids)
    return pred_points, pred_ids


def run_evaluation():
    device = torch.device(config.DEVICE)
    print(f"Loading Test Data from {config.DATA_ROOT}...")
    test_set = RadarFileDataset('test', include_empty=True)
    if len(test_set) == 0:
        return

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

    baseline_tracker = BaselineTracker(eps=POINT_CLUSTER_EPS, min_samples=POINT_CLUSTER_MIN_SAMPLES)
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
    enable_meas_diagnostic = ENABLE_MEAS_DIAGNOSTIC and gnn_model is not None
    enable_uncertainty_ablation = ENABLE_POINT_UNCERTAINTY_ABLATION and gnn_model is not None
    if enable_meas_diagnostic:
        metrics['H-GAT-GT (Meas Ablation)'] = create_metrics()
    if enable_uncertainty_ablation:
        metrics['H-GAT-GT (No Uncertainty Gating)'] = create_metrics()

    print('Running Point-Level Evaluation Loop (detected-only GT)...')
    for episode_idx in tqdm(range(len(test_set))):
        episode_graphs = test_set.get(episode_idx)

        gnn_group_tracker = GNNPostProcessor()
        gnn_meas_group_tracker = GNNPostProcessor() if enable_meas_diagnostic else None
        gnn_no_gate_group_tracker = GNNPostProcessor() if enable_uncertainty_ablation else None
        baseline_tracker.reset()
        gm_cphd_tracker.reset()
        cbmember_tracker.reset()
        graph_mb_tracker.reset()

        gnn_point_assoc = GroupConstrainedPointAssociator()
        gnn_meas_point_assoc = GroupConstrainedPointAssociator() if enable_meas_diagnostic else None
        gnn_no_gate_point_assoc = GroupConstrainedPointAssociator() if enable_uncertainty_ablation else None
        baseline_point_assoc = GroupConstrainedPointAssociator()
        gm_point_assoc = GroupConstrainedPointAssociator()
        cb_point_assoc = GroupConstrainedPointAssociator()
        gmb_point_assoc = GroupConstrainedPointAssociator()

        for metric in metrics.values():
            metric.reset_sequence()

        for frame_idx, graph in enumerate(episode_graphs):
            ensure_point_gt(graph, episode_idx, frame_idx)

            gt_pos, gt_ids = extract_detected_point_gt(graph, episode_idx, frame_idx)
            meas_points = graph.x.cpu().numpy()

            _, detected_centroids, centroid_to_points = cluster_measurements(meas_points)

            # H-GAT-GT
            t0 = time.time()
            corrected_pos = infer_corrected_points(gnn_model, graph, meas_points, device, head='point')
            filtered_corrected_pos, _, _ = filter_clustered_points(corrected_pos)
            pred_pos_gnn, pred_id_gnn = run_hgat_point_identity_pipeline(
                filtered_corrected_pos,
                gnn_group_tracker,
                gnn_point_assoc,
            )
            t1 = time.time()
            metrics['H-GAT-GT (Ours)'].update_time(t1 - t0)
            metrics['H-GAT-GT (Ours)'].update(gt_pos, gt_ids, pred_pos_gnn, pred_id_gnn)

            if enable_meas_diagnostic:
                t0 = time.time()
                filtered_meas_points, _, _ = filter_clustered_points(meas_points)
                pred_pos_gnn_meas, pred_id_gnn_meas = run_hgat_point_identity_pipeline(
                    filtered_meas_points,
                    gnn_meas_group_tracker,
                    gnn_meas_point_assoc,
                )
                t1 = time.time()
                metrics['H-GAT-GT (Meas Ablation)'].update_time(t1 - t0)
                metrics['H-GAT-GT (Meas Ablation)'].update(gt_pos, gt_ids, pred_pos_gnn_meas, pred_id_gnn_meas)

            if enable_uncertainty_ablation:
                t0 = time.time()
                corrected_pos_no_gate = infer_corrected_points(
                    gnn_model,
                    graph,
                    meas_points,
                    device,
                    head='point',
                    use_uncertainty_gating=False,
                )
                filtered_no_gate_pos, _, _ = filter_clustered_points(corrected_pos_no_gate)
                pred_pos_no_gate, pred_id_no_gate = run_hgat_point_identity_pipeline(
                    filtered_no_gate_pos,
                    gnn_no_gate_group_tracker,
                    gnn_no_gate_point_assoc,
                )
                t1 = time.time()
                metrics['H-GAT-GT (No Uncertainty Gating)'].update_time(t1 - t0)
                metrics['H-GAT-GT (No Uncertainty Gating)'].update(gt_pos, gt_ids, pred_pos_no_gate, pred_id_no_gate)

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
        'OSPA (Total)', 'OSPA (Loc)', 'OSPA (Card)', 'RMSE (Pos)', 'IDSW', 'Count Err',
        'MOTA', 'MOTP', 'FAR', 'Time'
    ]
    df = df[[c for c in cols if c in df.columns]]

    print('\n' + '=' * 100)
    print('FINAL POINT-LEVEL RESULTS (DETECTED-ONLY GT)')
    print('=' * 100)
    print('Note: point-level MOTA is auxiliary under detected-only GT; interpret OSPA / RMSE / IDSW first.')
    print(f'Prediction-side point filtering: DBSCAN eps={POINT_CLUSTER_EPS}, min_samples={POINT_CLUSTER_MIN_SAMPLES}; only non-noise points enter point tracking.')
    print(f'Point tracker thresholds: stage1={POINT_TRACK_STAGE1_THRESHOLD}, recovery={POINT_TRACK_RECOVERY_THRESHOLD}, max_age={POINT_TRACK_MAX_AGE}, metric_match={POINT_MATCH_THRESHOLD}.')
    print(f'Point IDSW gap tolerance: {POINT_ID_SWITCH_GAP_TOLERANCE} frame(s).')
    print('H-GAT-GT point IDs now use group tracking plus group-constrained point association.')
    if enable_meas_diagnostic:
        print('Diagnostic row included: H-GAT-GT (Meas Ablation) tracks filtered raw meas_points to compare against corrected_pos.')
    if enable_uncertainty_ablation:
        print('Ablation row included: H-GAT-GT (No Uncertainty Gating) disables point uncertainty gating with the same checkpoint.')
    print(df.to_string())
    print('=' * 100)


if __name__ == '__main__':
    run_evaluation()
