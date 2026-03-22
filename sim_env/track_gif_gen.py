import os
# 添加根目录路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError
from sklearn.metrics import accuracy_score

import config
from model import GNNGroupTracker
from dataset import RadarFileDataset


NUM = 6  # 选择第几个样本 episode

VIZ_STYLE = {
    'figsize': (8.6, 8.2),
    'dpi': 180,
    'fps': 10,
    'background': '#f4f2ee',
    'axes_face': '#fbfaf7',
    'grid_color': '#d7dde6',
    'grid_alpha': 0.55,
    'spine_color': '#c7ced8',
    'tick_color': '#556271',
    'title_color': '#1f2f43',
    'text_color': '#2e3a49',
    'clutter_color': '#aab3bf',
    'clutter_alpha': 0.28,
    'clutter_size': 12,
    'group_point_size': 44,
    'group_edge_width': 0.85,
    'group_edge_color': '#ffffff',
    'region_alpha': 0.14,
    'region_edge_alpha': 0.46,
    'region_edge_width': 1.2,
    'band_width': 13.0,
    'trail_alpha_min': 0.08,
    'trail_alpha_max': 0.52,
    'trail_width_min': 0.9,
    'trail_width_max': 2.6,
    'tail_length': 24,
    'centroid_outer_size': 132,
    'centroid_inner_size': 58,
    'label_fontsize': 9,
    'info_fontsize': 10,
    'margin_ratio': 0.08,
}

COLOR_PALETTE = [
    '#2F5D8C', '#D17A32', '#4D9A8A', '#8A5689', '#C55252',
    '#738B3A', '#3F7CB1', '#B86272', '#796AB2', '#C6A02B',
    '#2F8F9D', '#A86A3B'
]


# ==========================================
# 1. 升级版跟踪器: 匈牙利匹配 + 速度预测
# ==========================================
class RobustGroupTracker:
    def __init__(self, max_age=5, dist_thresh=60.0):
        self.tracks = {}  # {id: {'pos': [x,y], 'vel': [vx,vy], 'age': 0, 'trace': []}}
        self.next_id = 1
        self.max_age = max_age
        self.dist_thresh = dist_thresh

    def update(self, detected_centers):
        """
        detected_centers: 当前帧 GNN 聚类出的质心列表 [[x,y], ...]
        返回: {detection_idx: track_id}
        """
        # --- 1. 状态预测 (Predict) ---
        for tid, trk in self.tracks.items():
            trk['pos'] += trk['vel']
            trk['age'] += 1

        active_track_ids = list(self.tracks.keys())
        num_tracks = len(active_track_ids)
        num_dets = len(detected_centers)

        assignment = {}
        used_dets = set()
        used_tracks = set()

        # --- 2. 构建代价矩阵 (Cost Matrix) ---
        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.zeros((num_tracks, num_dets))
            for i, tid in enumerate(active_track_ids):
                pred_pos = self.tracks[tid]['pos']
                for j, det_pos in enumerate(detected_centers):
                    cost_matrix[i, j] = np.linalg.norm(pred_pos - det_pos)

            # --- 3. 匈牙利算法匹配 (Global Optimization) ---
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # --- 4. 过滤不合理的匹配 ---
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.dist_thresh:
                    tid = active_track_ids[r]
                    self._update_track_state(tid, detected_centers[c])
                    assignment[c] = tid
                    used_tracks.add(tid)
                    used_dets.add(c)

        # =========================================
        #   特殊事件处理：分裂与合并
        # =========================================

        # --- 检查分裂 (Split) ---
        for d_idx in range(num_dets):
            if d_idx in used_dets:
                continue

            det_pos = detected_centers[d_idx]
            best_dist = float('inf')
            parent_id = -1

            for tid in self.tracks:
                dist = np.linalg.norm(self.tracks[tid]['pos'] - det_pos)
                if dist < self.dist_thresh and dist < best_dist:
                    best_dist = dist
                    parent_id = tid

            if parent_id != -1:
                new_id = self.next_id
                self.next_id += 1
                self._create_track(new_id, det_pos, parent_vel=self.tracks[parent_id]['vel'])
                assignment[d_idx] = new_id
                used_dets.add(d_idx)

        # --- 检查合并 (Merge) ---
        for tid in active_track_ids:
            if tid in used_tracks:
                continue

            track_pos = self.tracks[tid]['pos']
            best_dist = float('inf')
            target_det_idx = -1

            for d_idx in range(num_dets):
                dist = np.linalg.norm(track_pos - detected_centers[d_idx])
                if dist < self.dist_thresh and dist < best_dist:
                    best_dist = dist
                    target_det_idx = d_idx

            if target_det_idx != -1:
                used_tracks.add(tid)

        # --- 处理新生目标 (New Birth) ---
        for d_idx in range(num_dets):
            if d_idx not in used_dets:
                new_id = self.next_id
                self.next_id += 1
                self._create_track(new_id, detected_centers[d_idx])
                assignment[d_idx] = new_id

        # --- 清理死亡目标 (Dead Tracks) ---
        to_delete = []
        for tid in self.tracks:
            if tid not in used_tracks and self.tracks[tid]['age'] > self.max_age:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return assignment

    def _create_track(self, tid, pos, parent_vel=None):
        vel = parent_vel.copy() if parent_vel is not None else np.zeros(2)
        self.tracks[tid] = {
            'pos': np.array(pos),
            'vel': np.array(vel),
            'age': 0,
            'trace': [np.array(pos)]
        }

    def _update_track_state(self, tid, curr_pos):
        alpha_pos = 0.6
        alpha_vel = 0.3

        prev_pos = self.tracks[tid]['pos']
        prev_vel = self.tracks[tid]['vel']

        inst_vel = curr_pos - prev_pos

        new_pos = prev_pos * (1 - alpha_pos) + curr_pos * alpha_pos
        new_vel = prev_vel * (1 - alpha_vel) + inst_vel * alpha_vel

        self.tracks[tid]['pos'] = new_pos
        self.tracks[tid]['vel'] = new_vel
        self.tracks[tid]['age'] = 0

        self.tracks[tid]['trace'].append(new_pos)
        if len(self.tracks[tid]['trace']) > 50:
            self.tracks[tid]['trace'].pop(0)


def get_track_color(track_id):
    return COLOR_PALETTE[(int(track_id) - 1) % len(COLOR_PALETTE)]


def compute_scene_limits(viz_frames):
    collected = []
    for data in viz_frames:
        if data['raw_pos'].size:
            collected.append(data['raw_pos'])
        if data['plot_pos'].size:
            collected.append(data['plot_pos'])
        if data['centers']:
            collected.append(np.vstack(list(data['centers'].values())))
        for trace in data['trace'].values():
            trace = np.asarray(trace)
            if trace.size:
                collected.append(trace)

    if not collected:
        return (-1.0, 1.0), (-1.0, 1.0)

    points = np.vstack(collected)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = 0.5 * (min_xy + max_xy)
    half_span = 0.5 * max(max_xy - min_xy)
    half_span = max(half_span, 20.0)
    half_span *= 1.0 + 2.0 * VIZ_STYLE['margin_ratio']

    return (center[0] - half_span, center[0] + half_span), (center[1] - half_span, center[1] + half_span)


def style_axes(ax):
    ax.set_facecolor(VIZ_STYLE['axes_face'])
    for spine in ax.spines.values():
        spine.set_color(VIZ_STYLE['spine_color'])
        spine.set_linewidth(0.9)

    ax.tick_params(colors=VIZ_STYLE['tick_color'], labelsize=9)
    ax.grid(True, linestyle=':', linewidth=0.8, color=VIZ_STYLE['grid_color'], alpha=VIZ_STYLE['grid_alpha'])
    ax.set_axisbelow(True)


def draw_group_region(ax, points, color):
    if len(points) < 2:
        return

    if len(points) == 2:
        ax.plot(
            points[:, 0], points[:, 1],
            color=mcolors.to_rgba(color, VIZ_STYLE['region_alpha'] + 0.04),
            linewidth=VIZ_STYLE['band_width'],
            solid_capstyle='round',
            zorder=1,
        )
        return

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        patch = Polygon(
            hull_points,
            closed=True,
            facecolor=mcolors.to_rgba(color, VIZ_STYLE['region_alpha']),
            edgecolor=mcolors.to_rgba(color, VIZ_STYLE['region_edge_alpha']),
            linewidth=VIZ_STYLE['region_edge_width'],
            joinstyle='round',
            zorder=1,
        )
        ax.add_patch(patch)
    except QhullError:
        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        start_idx, end_idx = np.unravel_index(np.argmax(dist), dist.shape)
        band = points[[start_idx, end_idx]]
        ax.plot(
            band[:, 0], band[:, 1],
            color=mcolors.to_rgba(color, VIZ_STYLE['region_alpha'] + 0.04),
            linewidth=VIZ_STYLE['band_width'],
            solid_capstyle='round',
            zorder=1,
        )


def draw_fading_trail(ax, trace, color):
    trace = np.asarray(trace)
    if len(trace) < 2:
        return

    trace = trace[-VIZ_STYLE['tail_length']:]
    num_segments = len(trace) - 1
    for idx in range(num_segments):
        frac = (idx + 1) / num_segments
        alpha = VIZ_STYLE['trail_alpha_min'] + frac * (VIZ_STYLE['trail_alpha_max'] - VIZ_STYLE['trail_alpha_min'])
        linewidth = VIZ_STYLE['trail_width_min'] + frac * (VIZ_STYLE['trail_width_max'] - VIZ_STYLE['trail_width_min'])
        ax.plot(
            trace[idx:idx + 2, 0], trace[idx:idx + 2, 1],
            color=mcolors.to_rgba(color, alpha),
            linewidth=linewidth,
            solid_capstyle='round',
            zorder=3,
        )


# ==========================================
# 2. 推理主流程 (带 Accuracy 计算)
# ==========================================
def run_inference_and_viz():
    device = torch.device(config.DEVICE)

    model = GNNGroupTracker().to(device)
    if not os.path.exists(config.MODEL_USE_PATH):
        print("Model not found! Train first.")
        return
    model.load_state_dict(torch.load(config.MODEL_USE_PATH, map_location=device))
    model.eval()

    test_set = RadarFileDataset('test')
    if len(test_set) == 0:
        print("Test set empty. Run generate_data.py.")
        return
    episode_graphs = test_set.get(NUM)

    tracker = RobustGroupTracker(dist_thresh=50.0)

    viz_frames = []
    print(f"Starting inference on {len(episode_graphs)} frames...")

    with torch.no_grad():
        for t, graph in enumerate(episode_graphs):
            graph = graph.to(device)
            raw_pos = graph.x.cpu().numpy()

            if graph.edge_index.shape[1] == 0:
                viz_frames.append({
                    'raw_pos': raw_pos,
                    'plot_pos': raw_pos.copy(),
                    'track_ids': np.full(graph.num_nodes, -1),
                    'centers': {},
                    'acc': 0.0,
                    'trace': {}
                })
                continue

            # --- Model Forward ---
            scores, offsets, _ = model(graph)

            # --- Calculate Accuracy ---
            preds = (scores > 0.5).float()
            edge_acc = accuracy_score(graph.edge_label.cpu().numpy(), preds.cpu().numpy())

            # --- Clustering ---
            mask = scores > 0.5
            edges = graph.edge_index[:, mask].cpu().numpy()
            num_nodes = graph.num_nodes

            if edges.shape[1] > 0:
                adj = coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(num_nodes, num_nodes))
                n_comps, labels = connected_components(adj, directed=False)
            else:
                n_comps = num_nodes
                labels = np.arange(num_nodes)

            # --- Extract Centroids ---
            offsets_np = offsets.cpu().numpy()
            corrected_pos = raw_pos + offsets_np

            clusters_centers = []
            cluster_id_map = []

            for cid in range(n_comps):
                idx = np.where(labels == cid)[0]
                if len(idx) < 3:
                    continue
                center = np.mean(corrected_pos[idx], axis=0)
                clusters_centers.append(center)
                cluster_id_map.append(cid)

            # --- Tracking ---
            assignment = tracker.update(clusters_centers)

            # --- Prepare Viz Data ---
            frame_data = {
                'raw_pos': raw_pos,
                'plot_pos': corrected_pos,
                'track_ids': np.full(num_nodes, -1),
                'centers': {},
                'acc': edge_acc,
                'trace': {}
            }

            for det_idx, track_id in assignment.items():
                original_cid = cluster_id_map[det_idx]
                idx = np.where(labels == original_cid)[0]
                frame_data['track_ids'][idx] = track_id
                frame_data['centers'][track_id] = clusters_centers[det_idx]
                frame_data['trace'][track_id] = np.array(tracker.tracks[track_id]['trace'])

            viz_frames.append(frame_data)
            print(f"Frame {t:02d} | Edge Acc: {edge_acc * 100:.2f}% | Active Tracks: {len(assignment)}")

    if not viz_frames:
        print("No frames to visualize.")
        return

    xlim, ylim = compute_scene_limits(viz_frames)

    print("Generating GIF...")
    fig, ax = plt.subplots(figsize=VIZ_STYLE['figsize'], dpi=VIZ_STYLE['dpi'])
    fig.patch.set_facecolor(VIZ_STYLE['background'])
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=0.92)

    def update(frame_idx):
        ax.clear()
        style_axes(ax)

        data = viz_frames[frame_idx]
        raw_pos = data['raw_pos']
        plot_pos = data['plot_pos']
        track_ids = data['track_ids']

        clutter_mask = track_ids == -1
        if np.any(clutter_mask):
            ax.scatter(
                raw_pos[clutter_mask, 0], raw_pos[clutter_mask, 1],
                color=VIZ_STYLE['clutter_color'],
                s=VIZ_STYLE['clutter_size'],
                alpha=VIZ_STYLE['clutter_alpha'],
                edgecolors='none',
                zorder=0,
            )

        unique_ids = [uid for uid in np.unique(track_ids) if uid != -1]
        for uid in unique_ids:
            mask = track_ids == uid
            color = get_track_color(uid)
            group_points = plot_pos[mask]

            draw_group_region(ax, group_points, color)

            if uid in data['trace']:
                draw_fading_trail(ax, data['trace'][uid], color)

            ax.scatter(
                group_points[:, 0], group_points[:, 1],
                color=color,
                s=VIZ_STYLE['group_point_size'],
                alpha=0.95,
                edgecolors=VIZ_STYLE['group_edge_color'],
                linewidths=VIZ_STYLE['group_edge_width'],
                zorder=5,
            )

            if uid in data['centers']:
                cx, cy = data['centers'][uid]
                ax.scatter(
                    cx, cy,
                    s=VIZ_STYLE['centroid_outer_size'],
                    color='white',
                    alpha=0.92,
                    edgecolors='none',
                    zorder=6,
                )
                ax.scatter(
                    cx, cy,
                    s=VIZ_STYLE['centroid_inner_size'],
                    color=color,
                    edgecolors='white',
                    linewidths=1.0,
                    zorder=7,
                )

                offset = (10, 9) if int(uid) % 2 else (10, -16)
                va = 'bottom' if int(uid) % 2 else 'top'
                ax.annotate(
                    f"G{int(uid)}",
                    xy=(cx, cy),
                    xytext=offset,
                    textcoords='offset points',
                    color=color,
                    fontsize=VIZ_STYLE['label_fontsize'],
                    fontweight='semibold',
                    ha='left',
                    va=va,
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        fc=mcolors.to_rgba('white', 0.88),
                        ec=mcolors.to_rgba(color, 0.55),
                        lw=0.8,
                    ),
                    zorder=8,
                )

        info_text = (
            f"Frame {frame_idx:02d}\n"
            f"Active groups  {len(data['centers'])}\n"
            f"Edge acc  {data['acc'] * 100:5.1f}%"
        )
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=VIZ_STYLE['info_fontsize'],
            color=VIZ_STYLE['text_color'],
            bbox=dict(boxstyle='round,pad=0.35', fc=mcolors.to_rgba('white', 0.80), ec='none'),
            zorder=10,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('H-GAT-GT Group Tracking', fontsize=14, color=VIZ_STYLE['title_color'], pad=12, fontweight='semibold')
        ax.set_xlabel('X (m)', fontsize=10, color=VIZ_STYLE['tick_color'])
        ax.set_ylabel('Y (m)', fontsize=10, color=VIZ_STYLE['tick_color'])

    os.makedirs(config.OUTPUT_GIF_DIR, exist_ok=True)
    path2 = os.path.join(config.OUTPUT_GIF_DIR, f"sim_data_{NUM}_track_result_paper.gif")
    ani = animation.FuncAnimation(fig, update, frames=len(viz_frames), interval=1000 // VIZ_STYLE['fps'])
    ani.save(path2, writer='pillow', fps=VIZ_STYLE['fps'], dpi=VIZ_STYLE['dpi'])
    plt.close(fig)
    print(f"Saved {path2}")


if __name__ == "__main__":
    run_inference_and_viz()
