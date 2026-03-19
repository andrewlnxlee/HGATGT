import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


POINT_TRACK_MAX_AGE = 20
POINT_TRACK_STAGE1_THRESHOLD = 14.0
POINT_TRACK_RECOVERY_THRESHOLD = 22.0


class GNNPointPostProcessor:
    def __init__(
        self,
        max_age=POINT_TRACK_MAX_AGE,
        stage1_thresh=POINT_TRACK_STAGE1_THRESHOLD,
        stage2_thresh=POINT_TRACK_RECOVERY_THRESHOLD,
    ):
        self.tracks = {}
        self.next_id = 1

        self.max_age = max_age
        self.stage1_thresh = stage1_thresh
        self.stage2_thresh = stage2_thresh

        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0, 0.5 * dt ** 2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt ** 2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        self.R = np.eye(2) * 25.0
        self.Q = np.diag([0.05, 0.05, 0.05, 0.05, 0.1, 0.1])
        self.P_init = np.diag([20.0, 20.0, 200.0, 200.0, 600.0, 600.0])

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detected_points):
        detected_points = np.asarray(detected_points, dtype=float).reshape(-1, 2)

        for trk in self.tracks.values():
            if trk['age'] > 0:
                trk['x'][4:] *= 0.5
                trk['x'][2:4] *= 0.9
            trk['x'] = self.F @ trk['x']
            trk['P'] = self.F @ trk['P'] @ self.F.T + self.Q
            trk['age'] += 1

        active_ids = list(self.tracks.keys())
        unmatched_dets = set(range(detected_points.shape[0]))
        unmatched_trks = set(active_ids)

        def associate(thresh, t_set, d_set):
            if not t_set or not d_set:
                return
            t_ids = list(t_set)
            d_ids = list(d_set)

            t_pos = np.array([self.tracks[tid]['x'][:2] for tid in t_ids])
            d_pos = detected_points[d_ids]
            dist_cost = cdist(t_pos, d_pos, metric='euclidean')
            row, col = linear_sum_assignment(dist_cost)

            for r, c in zip(row, col):
                if dist_cost[r, c] >= thresh:
                    continue
                tid = t_ids[r]
                did = d_ids[c]
                if self.tracks[tid]['age'] > 1:
                    self.tracks[tid]['P'] = self.P_init.copy()
                self._update_track(tid, detected_points[did])
                unmatched_trks.discard(tid)
                unmatched_dets.discard(did)

        associate(self.stage1_thresh, unmatched_trks, unmatched_dets)
        associate(self.stage2_thresh, unmatched_trks, unmatched_dets)

        for did in unmatched_dets:
            self._create_track(self.next_id, detected_points[did])
            self.next_id += 1

        out_c, out_id = [], []
        to_del = []
        for tid, trk in self.tracks.items():
            if trk['age'] == 0:
                out_c.append(trk['last_meas'])
                out_id.append(tid)
            if trk['age'] > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        return np.asarray(out_c).reshape(-1, 2), np.asarray(out_id, dtype=int)

    def _create_track(self, tid, pos):
        self.tracks[tid] = {
            'x': np.zeros(6),
            'P': self.P_init.copy(),
            'age': 0,
            'trace': [np.asarray(pos, dtype=float)],
            'last_meas': np.asarray(pos, dtype=float),
        }
        self.tracks[tid]['x'][:2] = pos

    def _update_track(self, tid, pos):
        trk = self.tracks[tid]
        y = pos - self.H @ trk['x']
        S = self.H @ trk['P'] @ self.H.T + self.R
        K = trk['P'] @ self.H.T @ np.linalg.inv(S)
        trk['x'] += K @ y
        trk['P'] = (np.eye(6) - K @ self.H) @ trk['P']
        trk['age'] = 0
        trk['last_meas'] = np.asarray(pos, dtype=float)
        trk['trace'].append(np.asarray(pos, dtype=float))
        if len(trk['trace']) > 50:
            trk['trace'].pop(0)
