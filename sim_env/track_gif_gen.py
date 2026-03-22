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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances

import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from trackers.gnn_processor import GNNPostProcessor


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
    'raw_group_size': 30,
    'raw_group_alpha': 0.42,
    'raw_group_edge_width': 1.15,
    'offset_line_alpha': 0.26,
    'offset_line_width': 0.9,
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


def unpack_group_offsets(model_out):
    if hasattr(model_out, 'get_offsets'):
        return model_out.get_offsets('group')
    if isinstance(model_out, (tuple, list)) and len(model_out) >= 2:
        return model_out[1]
    raise ValueError('模型输出缺少 group offset。')


def unpack_edge_scores(model_out):
    if hasattr(model_out, 'edge_scores'):
        return model_out.edge_scores
    if isinstance(model_out, (tuple, list)) and len(model_out) >= 1:
        return model_out[0]
    return None


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


def build_gnn_detections(raw_pos, corrected_pos):
    if len(corrected_pos) == 0:
        return np.empty((0, 2)), None, []

    try:
        labels = DBSCAN(eps=30, min_samples=3).fit(corrected_pos).labels_
    except Exception:
        labels = np.full(len(corrected_pos), -1)

    det_centers = []
    det_shapes = []
    cluster_indices_list = []

    for label in set(labels):
        if label == -1:
            continue
        indices = np.where(labels == label)[0]
        cluster_indices_list.append(indices)

        det_centers.append(np.mean(corrected_pos[indices], axis=0))

        pts_raw = raw_pos[indices]
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

    return det_centers, det_shapes, cluster_indices_list


def assign_tracks_to_points(pred_centers, pred_track_ids, det_centers, cluster_indices_list, num_nodes):
    point_track_ids = np.full(num_nodes, -1, dtype=int)
    centers = {}
    matched_track_ids = set()

    if len(pred_centers) == 0 or len(det_centers) == 0:
        return point_track_ids, centers, matched_track_ids

    cost_matrix = euclidean_distances(pred_centers, det_centers)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 20.0:
            track_id = int(pred_track_ids[r])
            point_indices = cluster_indices_list[c]
            point_track_ids[point_indices] = track_id
            centers[track_id] = pred_centers[r]
            matched_track_ids.add(track_id)

    return point_track_ids, centers, matched_track_ids


def compute_edge_accuracy(edge_scores, graph):
    if edge_scores is None or not hasattr(graph, 'edge_label'):
        return 0.0
    if graph.edge_label.numel() == 0:
        return 0.0

    preds = (edge_scores > 0.5).float()
    return accuracy_score(
        graph.edge_label.detach().cpu().numpy(),
        preds.detach().cpu().numpy(),
    )


# ==========================================
# 推理主流程（与 evaluate.py 主线对齐）
# ==========================================
def run_inference_and_viz():
    device = torch.device(config.DEVICE)

    model = GNNGroupTracker(
        input_node_dim=config.INPUT_DIM,
        input_edge_dim=config.EDGE_DIM,
        hidden_dim=config.HIDDEN_DIM,
    ).to(device)
    if not os.path.exists(config.MODEL_USE_PATH):
        print('Model not found! Train first.')
        return
    model.load_state_dict(torch.load(config.MODEL_USE_PATH, map_location=device))
    model.eval()

    test_set = RadarFileDataset('test')
    if len(test_set) == 0:
        print('Test set empty. Run generate_data.py.')
        return
    episode_graphs = test_set.get(NUM)

    gnn_processor = GNNPostProcessor()

    viz_frames = []
    print(f'Starting inference on {len(episode_graphs)} frames...')

    with torch.no_grad():
        for t, graph in enumerate(episode_graphs):
            graph_dev = graph.to(device)
            raw_pos = graph.x.detach().cpu().numpy()
            num_nodes = len(raw_pos)

            frame_data = {
                'raw_pos': raw_pos,
                'plot_pos': raw_pos.copy(),
                'track_ids': np.full(num_nodes, -1, dtype=int),
                'centers': {},
                'acc': 0.0,
                'trace': {},
            }

            if num_nodes == 0:
                gnn_processor.update(np.empty((0, 2)), None)
                viz_frames.append(frame_data)
                print(f'Frame {t:02d} | Edge Acc: 0.00% | Active Tracks: 0')
                continue

            model_out = model(graph_dev)
            edge_scores = unpack_edge_scores(model_out)
            group_offsets = unpack_group_offsets(model_out)
            frame_data['acc'] = compute_edge_accuracy(edge_scores, graph_dev)

            corrected_pos = raw_pos + group_offsets.detach().cpu().numpy()
            frame_data['plot_pos'] = corrected_pos

            det_centers, det_shapes, cluster_indices_list = build_gnn_detections(raw_pos, corrected_pos)

            if len(det_centers) > 0:
                pred_centers, pred_track_ids, _ = gnn_processor.update(det_centers, det_shapes)
            else:
                pred_centers, pred_track_ids, _ = gnn_processor.update(np.empty((0, 2)), None)

            pred_centers = np.array(pred_centers).reshape(-1, 2)
            pred_track_ids = np.array(pred_track_ids).reshape(-1)

            point_track_ids, centers, matched_track_ids = assign_tracks_to_points(
                pred_centers,
                pred_track_ids,
                det_centers,
                cluster_indices_list,
                num_nodes,
            )

            frame_data['track_ids'] = point_track_ids
            frame_data['centers'] = centers
            for track_id in matched_track_ids:
                if track_id in gnn_processor.tracks:
                    frame_data['trace'][track_id] = np.array(gnn_processor.tracks[track_id]['trace'])

            viz_frames.append(frame_data)
            print(f"Frame {t:02d} | Edge Acc: {frame_data['acc'] * 100:.2f}% | Active Tracks: {len(frame_data['centers'])}")

    if not viz_frames:
        print('No frames to visualize.')
        return

    xlim, ylim = compute_scene_limits(viz_frames)

    print('Generating GIF...')
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
            raw_group_points = raw_pos[mask]
            group_points = plot_pos[mask]

            draw_group_region(ax, group_points, color)

            if uid in data['trace']:
                draw_fading_trail(ax, data['trace'][uid], color)

            if len(raw_group_points):
                for raw_pt, fused_pt in zip(raw_group_points, group_points):
                    ax.plot(
                        [raw_pt[0], fused_pt[0]], [raw_pt[1], fused_pt[1]],
                        color=mcolors.to_rgba(color, VIZ_STYLE['offset_line_alpha']),
                        linewidth=VIZ_STYLE['offset_line_width'],
                        zorder=2,
                    )

                ax.scatter(
                    raw_group_points[:, 0], raw_group_points[:, 1],
                    facecolors='none',
                    edgecolors=mcolors.to_rgba(color, VIZ_STYLE['raw_group_alpha']),
                    s=VIZ_STYLE['raw_group_size'],
                    linewidths=VIZ_STYLE['raw_group_edge_width'],
                    zorder=4,
                )

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
                    f'G{int(uid)}',
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
            f'Frame {frame_idx:02d}\n'
            f'Active groups  {len(data["centers"])}\n'
            f'Edge acc  {data["acc"] * 100:5.1f}%'
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
    path2 = os.path.join(config.OUTPUT_GIF_DIR, f'sim_data_{NUM}_track_result_paper.gif')
    ani = animation.FuncAnimation(fig, update, frames=len(viz_frames), interval=max(1, 1000 // VIZ_STYLE['fps']))
    ani.save(path2, writer='pillow', fps=VIZ_STYLE['fps'], dpi=VIZ_STYLE['dpi'])
    plt.close(fig)
    print(f'Saved {path2}')


if __name__ == '__main__':
    run_inference_and_viz()
