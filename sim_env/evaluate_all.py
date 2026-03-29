import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import config
from dataset import RadarFileDataset
from metrics import TrackingMetrics
from model import GNNGroupTracker
from trackers.baseline import BaselineTracker
from trackers.cbmember import CBMeMBerTracker
from trackers.gm_cphd import GMCPHDTracker
from trackers.gnn_processor_all import (
    GROUP_TO_CLUSTER_THRESH,
    HierarchicalTrackProcessor,
    build_group_detections,
    compute_group_shape,
    project_cluster_tracks_to_points,
    assign_track_ids_to_points,
)
from trackers.graph_mb import GraphMBTracker


POINT_OSPA_C = 25.0
POINT_MATCH_THRESHOLD = float(getattr(config, 'POINT_MATCH_THRESHOLD', 35.0))
POINT_ID_SWITCH_GAP_TOLERANCE = int(getattr(config, 'POINT_ID_SWITCH_GAP_TOLERANCE', 3))
GROUP_ID_SWITCH_GAP_TOLERANCE = int(getattr(config, 'GROUP_ID_SWITCH_GAP_TOLERANCE', 3))
GROUP_MATCH_THRESHOLD = float(getattr(config, 'GROUP_MATCH_THRESHOLD', 40.0))
ENABLE_POINT_UNCERTAINTY_GATING = bool(getattr(config, 'ENABLE_POINT_UNCERTAINTY_GATING', False))


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


def build_shapes_from_point_labels(points, point_labels, target_ids):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    point_labels = np.asarray(point_labels, dtype=int).reshape(-1)
    target_ids = np.asarray(target_ids, dtype=int).reshape(-1)
    if len(target_ids) == 0:
        return None

    shapes = np.tile(np.array([3.0, 3.0]), (len(target_ids), 1))
    for i, target_id in enumerate(target_ids):
        idx = np.where(point_labels == target_id)[0]
        if len(idx) > 0:
            shapes[i] = compute_group_shape(points[idx])
    return shapes


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


def infer_dual_corrected_points(gnn_model, graph, meas_points, device):
    meas_points = np.asarray(meas_points, dtype=float).reshape(-1, 2)
    if gnn_model is None or len(meas_points) == 0 or graph.edge_index.shape[1] == 0:
        return meas_points, meas_points

    graph_dev = graph.to(device)
    with torch.no_grad():
        model_out = gnn_model(graph_dev)
        group_offsets, _ = unpack_head_outputs(model_out, head='group')
        point_offsets, point_uncertainty = unpack_head_outputs(model_out, head='point')

    group_offsets = group_offsets.detach().cpu().numpy().reshape(-1, 2)
    point_offsets = point_offsets.detach().cpu().numpy().reshape(-1, 2)
    if len(group_offsets) != len(meas_points) or len(point_offsets) != len(meas_points):
        raise ValueError(
            f'offset 数量与 meas 数量不一致: group={len(group_offsets)}, point={len(point_offsets)}, meas={len(meas_points)}'
        )

    if point_uncertainty is not None:
        point_uncertainty = point_uncertainty.detach().cpu().numpy().reshape(-1, 2)
        point_offsets = apply_point_uncertainty_gating(point_offsets, point_uncertainty)

    return meas_points + group_offsets, meas_points + point_offsets


def create_group_metrics():
    return TrackingMetrics(
        match_threshold=GROUP_MATCH_THRESHOLD,
        id_switch_gap_tolerance=GROUP_ID_SWITCH_GAP_TOLERANCE,
    )


def create_point_metrics():
    return TrackingMetrics(
        ospa_c=POINT_OSPA_C,
        match_threshold=POINT_MATCH_THRESHOLD,
        id_switch_gap_tolerance=POINT_ID_SWITCH_GAP_TOLERANCE,
    )


def update_group_metrics(metric, gt_centers, gt_ids, pred_centers, pred_ids, gt_shapes, pred_shapes, gt_point_labels, pred_point_group_ids):
    metric.update(
        gt_centers,
        gt_ids,
        pred_centers,
        pred_ids,
        gt_shapes=gt_shapes,
        pred_shapes=pred_shapes,
    )
    metric.update_clustering_metrics(gt_point_labels, pred_point_group_ids)


def update_point_metrics(metric, gt_pos, gt_ids, pred_pos, pred_ids):
    metric.update(gt_pos, gt_ids, pred_pos, pred_ids)


def build_pred_shapes_from_group_alignment(pred_group_centers, detected_centers, detected_shapes, match_thresh=GROUP_TO_CLUSTER_THRESH):
    pred_group_centers = np.asarray(pred_group_centers, dtype=float).reshape(-1, 2)
    detected_centers = np.asarray(detected_centers, dtype=float).reshape(-1, 2)
    if len(pred_group_centers) == 0:
        return None

    pred_shapes = np.tile(np.array([3.0, 3.0]), (len(pred_group_centers), 1))
    if detected_shapes is None or len(detected_centers) == 0:
        return pred_shapes

    detected_shapes = np.asarray(detected_shapes, dtype=float).reshape(-1, 2)
    cost = euclidean_distances(pred_group_centers, detected_centers)
    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < match_thresh:
            pred_shapes[r] = detected_shapes[c]
    return pred_shapes


def filter_clustered_points(points, eps=config.POINT_CLUSTER_EPS, min_samples=config.POINT_CLUSTER_MIN_SAMPLES):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    cluster_labels, _, _, _ = build_group_detections(points, eps=eps, min_samples=min_samples)
    keep_mask = cluster_labels != -1
    filtered_points = points[keep_mask]
    return filtered_points, keep_mask, cluster_labels


def run_hgat_tracker(processor, group_corrected_points, point_corrected_points, meas_points):
    group_corrected_points = np.asarray(group_corrected_points, dtype=float).reshape(-1, 2)
    point_corrected_points = np.asarray(point_corrected_points, dtype=float).reshape(-1, 2)
    meas_points = np.asarray(meas_points, dtype=float).reshape(-1, 2)

    group_out = processor.update_group_tracks(group_corrected_points)

    filtered_point_points, point_keep_mask, _ = filter_clustered_points(point_corrected_points)
    point_group_ids_for_points = group_out['point_group_ids'][point_keep_mask]

    point_out = processor.update_point_tracks(filtered_point_points, point_group_ids_for_points)

    point_group_ids_eval = assign_track_ids_to_points(
        group_out['group_centers'],
        group_out['group_ids'],
        group_out['detected_centers'],
        group_out['centroid_to_points'],
        len(meas_points),
        GROUP_TO_CLUSTER_THRESH,
    )
    detected_shapes_eval = None
    if len(group_out['centroid_to_points']) > 0:
        detected_shapes_eval = []
        for det_idx in sorted(group_out['centroid_to_points'].keys()):
            detected_shapes_eval.append(compute_group_shape(meas_points[group_out['centroid_to_points'][det_idx]]))
        detected_shapes_eval = np.asarray(detected_shapes_eval, dtype=float).reshape(-1, 2)

    pred_group_shapes_eval = build_pred_shapes_from_group_alignment(
        group_out['group_centers'],
        group_out['detected_centers'],
        detected_shapes_eval,
    )

    return {
        **group_out,
        **point_out,
        'point_group_ids_eval': point_group_ids_eval,
        'pred_group_shapes_eval': pred_group_shapes_eval,
    }


def run_baseline_tracker(group_tracker, point_tracker, meas_points):
    group_centers, group_ids, cluster_labels = group_tracker.step(meas_points)
    point_group_ids = project_cluster_tracks_to_points(meas_points, cluster_labels, group_centers, group_ids)
    point_positions, point_ids = point_tracker.update(meas_points, point_group_ids)
    group_shapes = build_shapes_from_point_labels(meas_points, point_group_ids, group_ids)
    return {
        'group_centers': np.asarray(group_centers, dtype=float).reshape(-1, 2),
        'group_ids': np.asarray(group_ids, dtype=int).reshape(-1),
        'group_shapes': group_shapes,
        'point_group_ids': point_group_ids,
        'point_positions': np.asarray(point_positions, dtype=float).reshape(-1, 2),
        'point_ids': np.asarray(point_ids, dtype=int).reshape(-1),
    }


def run_rfs_tracker(group_tracker, point_tracker, detected_centers, detected_shapes, centroid_to_points, meas_points):
    if len(detected_centers) == 0:
        group_centers, group_ids = group_tracker.step(np.empty((0, 2)))
    else:
        group_centers, group_ids = group_tracker.step(detected_centers)

    point_group_ids = np.full(len(meas_points), -1, dtype=int)
    detected_centers = np.asarray(detected_centers, dtype=float).reshape(-1, 2)
    group_centers = np.asarray(group_centers, dtype=float).reshape(-1, 2)
    group_ids = np.asarray(group_ids, dtype=int).reshape(-1)

    if len(detected_centers) > 0 and len(group_centers) > 0:
        cost = euclidean_distances(group_centers, detected_centers)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= GROUP_TO_CLUSTER_THRESH:
                continue
            if c in centroid_to_points:
                point_group_ids[centroid_to_points[c]] = group_ids[r]

    point_positions, point_ids = point_tracker.update(meas_points, point_group_ids)
    group_shapes = build_shapes_from_point_labels(meas_points, point_group_ids, group_ids)
    return {
        'group_centers': group_centers,
        'group_ids': group_ids,
        'group_shapes': group_shapes,
        'point_group_ids': point_group_ids,
        'point_positions': np.asarray(point_positions, dtype=float).reshape(-1, 2),
        'point_ids': np.asarray(point_ids, dtype=int).reshape(-1),
    }


def run_graph_mb_tracker(group_tracker, point_tracker, meas_points):
    group_centers, group_ids, point_group_ids = group_tracker.step(meas_points)
    point_positions, point_ids = point_tracker.update(meas_points, point_group_ids)
    group_shapes = build_shapes_from_point_labels(meas_points, point_group_ids, group_ids)
    return {
        'group_centers': np.asarray(group_centers, dtype=float).reshape(-1, 2),
        'group_ids': np.asarray(group_ids, dtype=int).reshape(-1),
        'group_shapes': group_shapes,
        'point_group_ids': np.asarray(point_group_ids, dtype=int).reshape(-1),
        'point_positions': np.asarray(point_positions, dtype=float).reshape(-1, 2),
        'point_ids': np.asarray(point_ids, dtype=int).reshape(-1),
    }


def print_results_table(title, results, columns):
    df = pd.DataFrame(results).T
    df = df[[c for c in columns if c in df.columns]]
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)
    print(df.to_string())
    print('=' * 100)


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

    metrics_group = {
        'Baseline (DBSCAN+KF)': create_group_metrics(),
        'GM-CPHD (Standard)': create_group_metrics(),
        'CBMeMBer (Standard)': create_group_metrics(),
        'Graph-MB (Paper)': create_group_metrics(),
        'H-GAT-GT (Ours)': create_group_metrics(),
    }
    metrics_point = {
        'Baseline (DBSCAN+KF)': create_point_metrics(),
        'GM-CPHD (Standard)': create_point_metrics(),
        'CBMeMBer (Standard)': create_point_metrics(),
        'Graph-MB (Paper)': create_point_metrics(),
        'H-GAT-GT (Ours)': create_point_metrics(),
    }

    print('Running Unified Group+Point Evaluation Loop...')
    for episode_idx in tqdm(range(len(test_set))):
        episode_graphs = test_set.get(episode_idx)

        hgat_processor = HierarchicalTrackProcessor()
        baseline_group_tracker = BaselineTracker(eps=config.POINT_CLUSTER_EPS, min_samples=config.POINT_CLUSTER_MIN_SAMPLES)
        baseline_point_tracker = HierarchicalTrackProcessor().point_tracker
        gm_group_tracker = GMCPHDTracker()
        gm_point_tracker = HierarchicalTrackProcessor().point_tracker
        cb_group_tracker = CBMeMBerTracker()
        cb_point_tracker = HierarchicalTrackProcessor().point_tracker
        gmb_group_tracker = GraphMBTracker()
        gmb_point_tracker = HierarchicalTrackProcessor().point_tracker

        for metric in metrics_group.values():
            metric.reset_sequence()
        for metric in metrics_point.values():
            metric.reset_sequence()

        for frame_idx, graph in enumerate(episode_graphs):
            ensure_point_gt(graph, episode_idx, frame_idx)

            gt_group_data = graph.gt_centers.cpu().numpy()
            gt_group_centers = gt_group_data[:, 1:3] if len(gt_group_data) > 0 else np.zeros((0, 2), dtype=float)
            gt_group_ids = gt_group_data[:, 0].astype(int) if len(gt_group_data) > 0 else np.zeros((0,), dtype=int)
            meas_points = graph.x.cpu().numpy()
            gt_point_labels = graph.point_labels.cpu().numpy().astype(int)
            gt_group_shapes = build_shapes_from_point_labels(meas_points, gt_point_labels, gt_group_ids)
            gt_point_pos, gt_point_ids = extract_detected_point_gt(graph, episode_idx, frame_idx)

            cluster_labels_meas, detected_centers_meas, centroid_to_points_meas, detected_shapes_meas = build_group_detections(
                meas_points,
                eps=config.POINT_CLUSTER_EPS,
                min_samples=config.POINT_CLUSTER_MIN_SAMPLES,
            )

            # H-GAT-GT
            t0 = time.time()
            group_corrected_points, point_corrected_points = infer_dual_corrected_points(gnn_model, graph, meas_points, device)
            hgat_out = run_hgat_tracker(hgat_processor, group_corrected_points, point_corrected_points, meas_points)
            t1 = time.time()
            metrics_group['H-GAT-GT (Ours)'].update_time(t1 - t0)
            metrics_point['H-GAT-GT (Ours)'].update_time(t1 - t0)
            update_group_metrics(
                metrics_group['H-GAT-GT (Ours)'],
                gt_group_centers,
                gt_group_ids,
                hgat_out['group_centers'],
                hgat_out['group_ids'],
                gt_group_shapes,
                hgat_out['pred_group_shapes_eval'],
                gt_point_labels,
                hgat_out['point_group_ids_eval'],
            )
            update_point_metrics(
                metrics_point['H-GAT-GT (Ours)'],
                gt_point_pos,
                gt_point_ids,
                hgat_out['point_positions'],
                hgat_out['point_ids'],
            )

            # Baseline
            t0 = time.time()
            baseline_out = run_baseline_tracker(baseline_group_tracker, baseline_point_tracker, meas_points)
            t1 = time.time()
            metrics_group['Baseline (DBSCAN+KF)'].update_time(t1 - t0)
            metrics_point['Baseline (DBSCAN+KF)'].update_time(t1 - t0)
            update_group_metrics(
                metrics_group['Baseline (DBSCAN+KF)'],
                gt_group_centers,
                gt_group_ids,
                baseline_out['group_centers'],
                baseline_out['group_ids'],
                gt_group_shapes,
                baseline_out['group_shapes'],
                gt_point_labels,
                baseline_out['point_group_ids'],
            )
            update_point_metrics(
                metrics_point['Baseline (DBSCAN+KF)'],
                gt_point_pos,
                gt_point_ids,
                baseline_out['point_positions'],
                baseline_out['point_ids'],
            )

            # GM-CPHD
            t0 = time.time()
            gm_out = run_rfs_tracker(
                gm_group_tracker,
                gm_point_tracker,
                detected_centers_meas,
                detected_shapes_meas,
                centroid_to_points_meas,
                meas_points,
            )
            t1 = time.time()
            metrics_group['GM-CPHD (Standard)'].update_time(t1 - t0)
            metrics_point['GM-CPHD (Standard)'].update_time(t1 - t0)
            update_group_metrics(
                metrics_group['GM-CPHD (Standard)'],
                gt_group_centers,
                gt_group_ids,
                gm_out['group_centers'],
                gm_out['group_ids'],
                gt_group_shapes,
                gm_out['group_shapes'],
                gt_point_labels,
                gm_out['point_group_ids'],
            )
            update_point_metrics(
                metrics_point['GM-CPHD (Standard)'],
                gt_point_pos,
                gt_point_ids,
                gm_out['point_positions'],
                gm_out['point_ids'],
            )

            # CBMeMBer
            t0 = time.time()
            cb_out = run_rfs_tracker(
                cb_group_tracker,
                cb_point_tracker,
                detected_centers_meas,
                detected_shapes_meas,
                centroid_to_points_meas,
                meas_points,
            )
            t1 = time.time()
            metrics_group['CBMeMBer (Standard)'].update_time(t1 - t0)
            metrics_point['CBMeMBer (Standard)'].update_time(t1 - t0)
            update_group_metrics(
                metrics_group['CBMeMBer (Standard)'],
                gt_group_centers,
                gt_group_ids,
                cb_out['group_centers'],
                cb_out['group_ids'],
                gt_group_shapes,
                cb_out['group_shapes'],
                gt_point_labels,
                cb_out['point_group_ids'],
            )
            update_point_metrics(
                metrics_point['CBMeMBer (Standard)'],
                gt_point_pos,
                gt_point_ids,
                cb_out['point_positions'],
                cb_out['point_ids'],
            )

            # Graph-MB
            t0 = time.time()
            gmb_out = run_graph_mb_tracker(gmb_group_tracker, gmb_point_tracker, meas_points)
            t1 = time.time()
            metrics_group['Graph-MB (Paper)'].update_time(t1 - t0)
            metrics_point['Graph-MB (Paper)'].update_time(t1 - t0)
            update_group_metrics(
                metrics_group['Graph-MB (Paper)'],
                gt_group_centers,
                gt_group_ids,
                gmb_out['group_centers'],
                gmb_out['group_ids'],
                gt_group_shapes,
                gmb_out['group_shapes'],
                gt_point_labels,
                gmb_out['point_group_ids'],
            )
            update_point_metrics(
                metrics_point['Graph-MB (Paper)'],
                gt_point_pos,
                gt_point_ids,
                gmb_out['point_positions'],
                gmb_out['point_ids'],
            )

    group_results = {name: metric.compute() for name, metric in metrics_group.items()}
    point_results = {name: metric.compute() for name, metric in metrics_point.items()}

    print_results_table(
        'FINAL UNIFIED GROUP-LEVEL RESULTS',
        group_results,
        [
            'MOTA', 'MOTP', 'IDSW', 'FAR', 'OSPA (Total)', 'OSPA (Loc)', 'OSPA (Card)',
            'RMSE (Pos)', 'Count Err', 'Purity', 'Comp', 'G-IoU', 'Time',
        ],
    )
    print_results_table(
        'FINAL UNIFIED POINT-LEVEL RESULTS (DETECTED-ONLY GT)',
        point_results,
        [
            'OSPA (Total)', 'OSPA (Loc)', 'OSPA (Card)', 'RMSE (Pos)', 'IDSW',
            'Count Err', 'MOTA', 'MOTP', 'FAR', 'Time',
        ],
    )


if __name__ == '__main__':
    run_evaluation()
