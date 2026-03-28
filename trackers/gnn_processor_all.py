import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

import config
from trackers.gnn_processor import GNNPostProcessor


GROUP_CLUSTER_EPS = float(getattr(config, 'GROUP_CLUSTER_EPS', getattr(config, 'POINT_CLUSTER_EPS', 35.0)))
GROUP_CLUSTER_MIN_SAMPLES = int(getattr(config, 'GROUP_CLUSTER_MIN_SAMPLES', getattr(config, 'POINT_CLUSTER_MIN_SAMPLES', 3)))
GROUP_TO_CLUSTER_THRESH = float(getattr(config, 'GROUP_TO_CLUSTER_THRESH', getattr(config, 'GROUP_TO_CENTROID_THRESH', 20.0)))
POINT_TRACK_STAGE1_THRESHOLD = float(getattr(config, 'POINT_TRACK_STAGE1_THRESHOLD', 20.0))
POINT_TRACK_RECOVERY_THRESHOLD = float(getattr(config, 'POINT_TRACK_RECOVERY_THRESHOLD', 28.0))
POINT_TRACK_MAX_AGE = int(getattr(config, 'POINT_TRACK_MAX_AGE', 15))
GROUP_POINT_MAX_AGE = int(getattr(config, 'GROUP_POINT_MAX_AGE', min(POINT_TRACK_MAX_AGE, 4)))


def compute_group_shape(points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) > 1:
        lower = np.percentile(points, 5, axis=0)
        upper = np.percentile(points, 95, axis=0)
        wh = upper - lower
    else:
        wh = np.zeros(2, dtype=float)
    return np.maximum(wh, 3.0)


def build_group_detections(points, eps=GROUP_CLUSTER_EPS, min_samples=GROUP_CLUSTER_MIN_SAMPLES):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, 2), dtype=float), {}, None

    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    detected_centers = []
    detected_shapes = []
    centroid_to_points = {}

    valid_labels = sorted(label for label in np.unique(cluster_labels) if label != -1)
    for det_idx, label in enumerate(valid_labels):
        point_indices = np.where(cluster_labels == label)[0]
        detected_centers.append(np.mean(points[point_indices], axis=0))
        detected_shapes.append(compute_group_shape(points[point_indices]))
        centroid_to_points[det_idx] = point_indices

    detected_centers = np.asarray(detected_centers, dtype=float).reshape(-1, 2)
    detected_shapes = np.asarray(detected_shapes, dtype=float).reshape(-1, 2)
    if len(detected_shapes) == 0:
        detected_shapes = None
    return cluster_labels, detected_centers, centroid_to_points, detected_shapes


def assign_track_ids_to_points(track_centers, track_ids, detected_centers, centroid_to_points, num_points, dist_thresh=GROUP_TO_CLUSTER_THRESH):
    point_track_ids = np.full(int(num_points), -1, dtype=int)
    track_centers = np.asarray(track_centers, dtype=float).reshape(-1, 2)
    track_ids = np.asarray(track_ids, dtype=int).reshape(-1)
    detected_centers = np.asarray(detected_centers, dtype=float).reshape(-1, 2)

    if len(track_centers) == 0 or len(detected_centers) == 0:
        return point_track_ids

    cost = euclidean_distances(track_centers, detected_centers)
    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= dist_thresh:
            continue
        if c in centroid_to_points:
            point_track_ids[centroid_to_points[c]] = track_ids[r]
    return point_track_ids


def project_cluster_tracks_to_points(points, cluster_labels, track_centers, track_ids, dist_thresh=GROUP_TO_CLUSTER_THRESH):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    cluster_labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    point_track_ids = np.full(len(points), -1, dtype=int)

    valid_labels = sorted(label for label in np.unique(cluster_labels) if label != -1)
    if len(valid_labels) == 0:
        return point_track_ids

    detected_centers = []
    centroid_to_points = {}
    for det_idx, label in enumerate(valid_labels):
        point_indices = np.where(cluster_labels == label)[0]
        detected_centers.append(np.mean(points[point_indices], axis=0))
        centroid_to_points[det_idx] = point_indices

    detected_centers = np.asarray(detected_centers, dtype=float).reshape(-1, 2)
    return assign_track_ids_to_points(track_centers, track_ids, detected_centers, centroid_to_points, len(points), dist_thresh)


class GroupConstrainedPointTracker:
    def __init__(
        self,
        max_age=GROUP_POINT_MAX_AGE,
        stage1_thresh=POINT_TRACK_STAGE1_THRESHOLD,
        stage2_thresh=POINT_TRACK_RECOVERY_THRESHOLD,
    ):
        self.max_age = max_age
        self.stage1_thresh = stage1_thresh
        self.stage2_thresh = stage2_thresh
        self.match_threshold = stage2_thresh
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, points, group_ids):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        group_ids = np.asarray(group_ids, dtype=int).reshape(-1)
        if len(points) != len(group_ids):
            raise ValueError(f'points/group_ids 数量不一致: {len(points)} vs {len(group_ids)}')

        for trk in self.tracks.values():
            trk['pred_pos'] = trk['pos'] + trk['vel']
            trk['age'] += 1

        valid_mask = group_ids >= 0
        points = points[valid_mask]
        group_ids = group_ids[valid_mask]
        valid_group_ids = sorted(int(gid) for gid in np.unique(group_ids) if gid >= 0)
        for gid in valid_group_ids:
            det_indices = np.where(group_ids == gid)[0]
            if len(det_indices) == 0:
                continue

            track_ids = [tid for tid, trk in self.tracks.items() if trk['group_id'] == gid]
            matched_det_indices = set()

            if track_ids:
                pred_pos = np.array([self.tracks[tid]['pred_pos'] for tid in track_ids], dtype=float).reshape(-1, 2)
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

        active_ids = sorted(tid for tid, trk in self.tracks.items() if trk['age'] == 0)
        if not active_ids:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)

        pred_points = np.array([self.tracks[tid]['pos'] for tid in active_ids], dtype=float).reshape(-1, 2)
        pred_ids = np.array(active_ids, dtype=int)
        return pred_points, pred_ids


class HierarchicalTrackProcessor:
    def __init__(
        self,
        group_cluster_eps=GROUP_CLUSTER_EPS,
        group_cluster_min_samples=GROUP_CLUSTER_MIN_SAMPLES,
        group_to_cluster_thresh=GROUP_TO_CLUSTER_THRESH,
    ):
        self.group_cluster_eps = group_cluster_eps
        self.group_cluster_min_samples = group_cluster_min_samples
        self.group_to_cluster_thresh = group_to_cluster_thresh

        self.group_tracker = GNNPostProcessor()
        self.point_tracker = GroupConstrainedPointTracker()

    def reset(self):
        self.group_tracker = GNNPostProcessor()
        self.point_tracker.reset()

    def build_group_detections(self, corrected_points):
        return build_group_detections(corrected_points, self.group_cluster_eps, self.group_cluster_min_samples)

    def update_group_tracks(self, group_corrected_points):
        cluster_labels, detected_centers, centroid_to_points, detected_shapes = self.build_group_detections(group_corrected_points)
        if len(detected_centers) > 0:
            group_centers, group_ids, group_shapes = self.group_tracker.update(detected_centers, detected_shapes)
        else:
            group_centers, group_ids, group_shapes = self.group_tracker.update(np.empty((0, 2)), None)

        point_group_ids = assign_track_ids_to_points(
            group_centers,
            group_ids,
            detected_centers,
            centroid_to_points,
            len(group_corrected_points),
            self.group_to_cluster_thresh,
        )
        return {
            'cluster_labels': cluster_labels,
            'detected_centers': detected_centers,
            'detected_shapes': detected_shapes,
            'centroid_to_points': centroid_to_points,
            'group_centers': np.asarray(group_centers, dtype=float).reshape(-1, 2),
            'group_ids': np.asarray(group_ids, dtype=int).reshape(-1),
            'group_shapes': None if group_shapes is None else np.asarray(group_shapes, dtype=float).reshape(-1, 2),
            'point_group_ids': point_group_ids,
        }

    def update_point_tracks(self, point_corrected_points, point_group_ids):
        point_positions, point_ids = self.point_tracker.update(point_corrected_points, point_group_ids)
        return {
            'point_positions': np.asarray(point_positions, dtype=float).reshape(-1, 2),
            'point_ids': np.asarray(point_ids, dtype=int).reshape(-1),
        }

    def update_all(self, group_corrected_points, point_corrected_points):
        group_corrected_points = np.asarray(group_corrected_points, dtype=float).reshape(-1, 2)
        point_corrected_points = np.asarray(point_corrected_points, dtype=float).reshape(-1, 2)
        if len(group_corrected_points) != len(point_corrected_points):
            raise ValueError(
                f'group_corrected_points / point_corrected_points 数量不一致: '
                f'{len(group_corrected_points)} vs {len(point_corrected_points)}'
            )

        group_out = self.update_group_tracks(group_corrected_points)
        point_out = self.update_point_tracks(point_corrected_points, group_out['point_group_ids'])
        return {**group_out, **point_out}
