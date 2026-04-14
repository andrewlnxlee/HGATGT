import os
# 添加根目录路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib import font_manager
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


NUM = 20  # 选择第几个样本 episode

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
    'tail_length': None,
    'ghost_age': 3,
    'ghost_alpha_scale': 0.62,
    'inherit_dist_thresh': 26.0,
    'inherit_history_frames': 4,
    'bridge_frames': 6,
    'bridge_alpha': 0.52,
    'bridge_width': 1.5,
    'merge_marker_outer_size': 84,
    'merge_marker_inner_size': 42,
    'merge_bridge_width_boost': 1.1,
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

CJK_FONT_CANDIDATES = [
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans SC', 'Source Han Sans SC',
    'Source Han Sans CN', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei',
    'PingFang SC', 'Heiti SC', 'Arial Unicode MS'
]


def configure_text_labels():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CJK_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name] + list(plt.rcParams.get('font.sans-serif', []))
            plt.rcParams['axes.unicode_minus'] = False
            return {
                'title': 'H-GAT-GT 群组轨迹跟踪',
                'xlabel': 'X (m)',
                'ylabel': 'Y (m)',
                'frame': '帧号',
                'active_groups': '活跃群组',
            }

    plt.rcParams['axes.unicode_minus'] = False
    return {
        'title': 'H-GAT-GT Group Tracking',
        'xlabel': 'X (m)',
        'ylabel': 'Y (m)',
        'frame': 'Frame',
        'active_groups': 'Active groups',
    }


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


def adjust_color(color, factor):
    rgb = np.array(mcolors.to_rgb(color))
    if factor >= 1.0:
        return tuple(np.clip(rgb + (1.0 - rgb) * (factor - 1.0), 0.0, 1.0))
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def build_display_trace(base_trace, inherited_prefix):
    trace_parts = []
    if len(inherited_prefix):
        trace_parts.append(np.asarray(inherited_prefix, dtype=float))
    if len(base_trace):
        base_arr = np.asarray(base_trace, dtype=float)
        if trace_parts and np.allclose(trace_parts[-1][-1], base_arr[0]):
            base_arr = base_arr[1:]
        if len(base_arr):
            trace_parts.append(base_arr)

    if not trace_parts:
        return np.empty((0, 2), dtype=float)

    return np.vstack(trace_parts)


def compute_display_tracks(gnn_processor, lineage_meta, bridge_events):
    display_tracks = {}
    for track_id, trk in gnn_processor.tracks.items():
        root_id = lineage_meta.get(track_id, {}).get('root_id', track_id)
        generation = lineage_meta.get(track_id, {}).get('generation', 0)
        base_color = get_track_color(root_id)
        color_factor = 1.0 + min(generation, 3) * 0.08
        trace = build_display_trace(
            trk.get('trace', []),
            lineage_meta.get(track_id, {}).get('inherited_prefix', np.empty((0, 2), dtype=float)),
        )
        display_tracks[track_id] = {
            'root_id': root_id,
            'generation': generation,
            'parents': tuple(lineage_meta.get(track_id, {}).get('parents', ())),
            'trace': trace,
            'age': int(trk.get('age', 0)),
            'is_observed': int(trk.get('age', 0)) == 0,
            'center': np.asarray(trk.get('last_meas', trk['x'][:2]), dtype=float),
            'pred_center': np.asarray(trk['x'][:2], dtype=float),
            'color': adjust_color(base_color, color_factor),
            'shape': np.asarray(trk.get('shape', np.array([3.0, 3.0])), dtype=float),
        }

    for event in bridge_events:
        child_id = event['child_id']
        if child_id in display_tracks:
            display_tracks[child_id].setdefault('bridges', []).append(event)

    return display_tracks


def update_recent_history(recent_history, display_tracks, frame_idx):
    for track_id, info in display_tracks.items():
        recent_history.setdefault(track_id, []).append({
            'frame_idx': frame_idx,
            'center': np.asarray(info['center'], dtype=float),
            'trace': np.asarray(info['trace'], dtype=float),
            'root_id': info['root_id'],
            'generation': info['generation'],
        })
        recent_history[track_id] = recent_history[track_id][-VIZ_STYLE['inherit_history_frames']:]


def create_lineage_events(new_track_ids, prev_track_ids, display_tracks, recent_history, frame_idx):
    events = []
    if not new_track_ids or not prev_track_ids:
        return events

    child_candidates = []
    for child_id in new_track_ids:
        child_info = display_tracks.get(child_id)
        if child_info is None:
            continue
        child_center = np.asarray(child_info['center'], dtype=float)
        candidates = []
        for parent_id in prev_track_ids:
            history = recent_history.get(parent_id)
            if not history:
                continue
            parent_snapshot = history[-1]
            parent_center = np.asarray(parent_snapshot['center'], dtype=float)
            distance = float(np.linalg.norm(child_center - parent_center))
            if distance <= VIZ_STYLE['inherit_dist_thresh']:
                candidates.append((distance, parent_id, parent_snapshot))
        if candidates:
            child_candidates.append((child_id, sorted(candidates, key=lambda item: item[0])))

    if not child_candidates:
        return events

    parent_children = {}
    for child_id, candidates in child_candidates:
        _, best_parent, best_snapshot = candidates[0]
        parent_children.setdefault(best_parent, []).append((child_id, best_snapshot))

    split_parents = {pid for pid, childs in parent_children.items() if len(childs) > 1}

    child_all_candidates = {child_id: candidates for child_id, candidates in child_candidates}
    merge_children = set()
    for child_id, candidates in child_all_candidates.items():
        close_candidates = [item for item in candidates if item[0] <= VIZ_STYLE['inherit_dist_thresh']]
        if len(close_candidates) > 1:
            merge_children.add(child_id)

    handled_pairs = set()
    for child_id in sorted(merge_children):
        candidates = child_all_candidates[child_id]
        parents = [item[1] for item in candidates]
        parent_snapshots = {item[1]: item[2] for item in candidates}
        primary_parent = candidates[0][1]
        primary_snapshot = parent_snapshots[primary_parent]
        events.append({
            'type': 'merge',
            'child_id': child_id,
            'parents': parents,
            'primary_parent': primary_parent,
            'root_id': primary_snapshot['root_id'],
            'generation': primary_snapshot['generation'] + 1,
            'prefix': np.asarray(primary_snapshot['trace'], dtype=float),
            'bridges': [
                {
                    'start': np.asarray(parent_snapshots[parent_id]['center'], dtype=float),
                    'end': np.asarray(display_tracks[child_id]['center'], dtype=float),
                    'start_frame': frame_idx,
                    'type': 'merge',
                }
                for parent_id in parents
            ],
        })
        handled_pairs.add((primary_parent, child_id))

    for parent_id in sorted(split_parents):
        children = parent_children[parent_id]
        parent_snapshot = children[0][1]
        for child_id, _ in children:
            if child_id in merge_children:
                continue
            events.append({
                'type': 'split',
                'child_id': child_id,
                'parents': [parent_id],
                'primary_parent': parent_id,
                'root_id': parent_snapshot['root_id'],
                'generation': parent_snapshot['generation'] + 1,
                'prefix': np.asarray(parent_snapshot['trace'], dtype=float),
                'bridges': [{
                    'start': np.asarray(parent_snapshot['center'], dtype=float),
                    'end': np.asarray(display_tracks[child_id]['center'], dtype=float),
                    'start_frame': frame_idx,
                    'type': 'split',
                }],
            })
            handled_pairs.add((parent_id, child_id))

    for child_id, candidates in child_candidates:
        if child_id in merge_children:
            continue
        best_distance, best_parent, best_snapshot = candidates[0]
        if (best_parent, child_id) in handled_pairs:
            continue
        events.append({
            'type': 'inherit',
            'child_id': child_id,
            'parents': [best_parent],
            'primary_parent': best_parent,
            'root_id': best_snapshot['root_id'],
            'generation': best_snapshot['generation'] + 1,
            'prefix': np.asarray(best_snapshot['trace'], dtype=float),
            'bridges': [{
                'start': np.asarray(best_snapshot['center'], dtype=float),
                'end': np.asarray(display_tracks[child_id]['center'], dtype=float),
                'start_frame': frame_idx,
                'type': 'inherit',
            }],
        })

    deduped = {}
    for event in events:
        child_id = event['child_id']
        priority = {'merge': 0, 'split': 1, 'inherit': 2}[event['type']]
        existing = deduped.get(child_id)
        if existing is None or priority < existing[0]:
            deduped[child_id] = (priority, event)
    return [item[1] for item in deduped.values()]


def apply_lineage_events(lineage_meta, bridge_events, events):
    for event in events:
        child_id = event['child_id']
        lineage_meta[child_id] = {
            'root_id': event['root_id'],
            'parents': tuple(event['parents']),
            'generation': event['generation'],
            'inherited_prefix': np.asarray(event['prefix'], dtype=float),
        }
        bridge_events.extend(event['bridges'])


def compute_scene_limits(viz_frames):
    collected = []
    for data in viz_frames:
        if data['raw_pos'].size:
            collected.append(data['raw_pos'])
        if data['plot_pos'].size:
            collected.append(data['plot_pos'])
        if data['centers']:
            collected.append(np.vstack(list(data['centers'].values())))
        for track_info in data.get('display_tracks', {}).values():
            trace = np.asarray(track_info.get('trace', []))
            if trace.size:
                collected.append(trace)
            for bridge in track_info.get('bridges', []):
                collected.append(np.vstack([bridge['start'], bridge['end']]))

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


def draw_fading_trail(ax, trace, color, alpha_scale=1.0):
    trace = np.asarray(trace)
    if len(trace) < 2:
        return

    tail_length = VIZ_STYLE['tail_length']
    if tail_length is not None:
        trace = trace[-tail_length:]

    num_segments = len(trace) - 1
    for idx in range(num_segments):
        frac = (idx + 1) / num_segments
        alpha = VIZ_STYLE['trail_alpha_min'] + frac * (VIZ_STYLE['trail_alpha_max'] - VIZ_STYLE['trail_alpha_min'])
        linewidth = VIZ_STYLE['trail_width_min'] + frac * (VIZ_STYLE['trail_width_max'] - VIZ_STYLE['trail_width_min'])
        ax.plot(
            trace[idx:idx + 2, 0], trace[idx:idx + 2, 1],
            color=mcolors.to_rgba(color, min(alpha * alpha_scale, 0.95)),
            linewidth=linewidth,
            solid_capstyle='round',
            zorder=3,
        )


def draw_bridge(ax, bridge, color, frame_idx):
    age = frame_idx - bridge['start_frame']
    if age < 0 or age >= VIZ_STYLE['bridge_frames']:
        return

    fade = 1.0 - age / max(VIZ_STYLE['bridge_frames'], 1)
    ax.plot(
        [bridge['start'][0], bridge['end'][0]],
        [bridge['start'][1], bridge['end'][1]],
        color=mcolors.to_rgba(color, VIZ_STYLE['bridge_alpha'] * fade),
        linewidth=VIZ_STYLE['bridge_width'],
        linestyle='--',
        solid_capstyle='round',
        zorder=2,
    )


def collect_overview_tracks(viz_frames):
    overview_tracks = {}
    overview_bridges = {}

    for data in viz_frames:
        for track_id, track_info in data.get('display_tracks', {}).items():
            trace = np.asarray(track_info.get('trace', []), dtype=float)
            if not trace.size:
                continue
            existing = overview_tracks.get(track_id)
            if existing is None or len(trace) >= len(existing['trace']):
                overview_tracks[track_id] = {
                    'trace': trace.copy(),
                    'color': track_info['color'],
                    'root_id': track_info['root_id'],
                }

            for bridge in track_info.get('bridges', []):
                key = (
                    bridge['start_frame'],
                    tuple(np.round(bridge['start'], 4)),
                    tuple(np.round(bridge['end'], 4)),
                    bridge['type'],
                )
                overview_bridges[key] = {
                    'start': np.asarray(bridge['start'], dtype=float),
                    'end': np.asarray(bridge['end'], dtype=float),
                    'type': bridge['type'],
                    'child_id': track_id,
                    'start_frame': bridge['start_frame'],
                }

    return overview_tracks, list(overview_bridges.values())


def save_track_overview(viz_frames, xlim, ylim, labels):
    overview_tracks, overview_bridges = collect_overview_tracks(viz_frames)
    if not overview_tracks:
        return None

    fig, ax = plt.subplots(figsize=VIZ_STYLE['figsize'], dpi=VIZ_STYLE['dpi'])
    fig.patch.set_facecolor(VIZ_STYLE['background'])
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=0.97)
    style_axes(ax)

    merge_endpoints = {}
    for bridge in overview_bridges:
        child_track = overview_tracks.get(bridge['child_id'])
        color = child_track['color'] if child_track is not None else get_track_color(bridge['child_id'])
        bridge_alpha = VIZ_STYLE['bridge_alpha'] * (0.95 if bridge['type'] == 'merge' else 0.80)
        bridge_width = VIZ_STYLE['bridge_width'] + (VIZ_STYLE['merge_bridge_width_boost'] if bridge['type'] == 'merge' else 0.0)
        if bridge['type'] == 'merge':
            ax.annotate(
                '',
                xy=(bridge['end'][0], bridge['end'][1]),
                xytext=(bridge['start'][0], bridge['start'][1]),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=mcolors.to_rgba(color, bridge_alpha),
                    lw=bridge_width,
                    linestyle='--',
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=14,
                ),
                zorder=1,
            )
        else:
            ax.plot(
                [bridge['start'][0], bridge['end'][0]],
                [bridge['start'][1], bridge['end'][1]],
                color=mcolors.to_rgba(color, bridge_alpha),
                linewidth=bridge_width,
                linestyle='--',
                solid_capstyle='round',
                zorder=1,
            )
        if bridge['type'] == 'merge':
            key = tuple(np.round(bridge['end'], 4))
            merge_endpoints.setdefault(key, {
                'point': np.asarray(bridge['end'], dtype=float),
                'color': color,
                'count': 0,
            })
            merge_endpoints[key]['count'] += 1

    for track_id in sorted(overview_tracks):
        track_info = overview_tracks[track_id]
        trace = track_info['trace']
        color = track_info['color']
        ax.plot(
            trace[:, 0], trace[:, 1],
            color=mcolors.to_rgba(color, 0.34),
            linewidth=1.6,
            solid_capstyle='round',
            zorder=2,
        )
        ax.scatter(
            trace[:, 0], trace[:, 1],
            color=mcolors.to_rgba(color, 0.52),
            s=10,
            edgecolors='none',
            zorder=3,
        )
        ax.scatter(
            trace[0, 0], trace[0, 1],
            color=mcolors.to_rgba(color, 0.78),
            s=28,
            marker='o',
            edgecolors='white',
            linewidths=0.7,
            zorder=4,
        )
        ax.scatter(
            trace[-1, 0], trace[-1, 1],
            color=color,
            s=40,
            marker='X',
            edgecolors='white',
            linewidths=0.7,
            zorder=5,
        )
        ax.annotate(
            f'G{int(track_id)}',
            xy=(trace[-1, 0], trace[-1, 1]),
            xytext=(8, 6),
            textcoords='offset points',
            color=color,
            fontsize=VIZ_STYLE['label_fontsize'],
            fontweight='semibold',
            ha='left',
            va='bottom',
            bbox=dict(
                boxstyle='round,pad=0.25',
                fc=mcolors.to_rgba('white', 0.88),
                ec=mcolors.to_rgba(color, 0.55),
                lw=0.8,
            ),
            zorder=6,
        )

    for merge_info in merge_endpoints.values():
        if merge_info['count'] < 2:
            continue
        point = merge_info['point']
        color = merge_info['color']
        ax.scatter(
            point[0], point[1],
            s=VIZ_STYLE['merge_marker_outer_size'],
            color='white',
            edgecolors='none',
            alpha=0.92,
            zorder=6,
        )
        ax.scatter(
            point[0], point[1],
            s=VIZ_STYLE['merge_marker_inner_size'],
            color=color,
            marker='P',
            edgecolors='white',
            linewidths=0.8,
            zorder=7,
        )

    info_text = (
        f'{labels["frame"]}: 00-{len(viz_frames) - 1:02d}\n'
        f'{labels["active_groups"]}: {len(overview_tracks)}'
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
    ax.set_xlabel(labels['xlabel'], fontsize=10, color=VIZ_STYLE['tick_color'])
    ax.set_ylabel(labels['ylabel'], fontsize=10, color=VIZ_STYLE['tick_color'])

    os.makedirs(config.OUTPUT_GIF_DIR, exist_ok=True)
    overview_path = os.path.join(config.OUTPUT_GIF_DIR, f'sim_data_{NUM}_track_overview_paper.png')
    fig.savefig(overview_path, dpi=VIZ_STYLE['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return overview_path


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
    labels = configure_text_labels()
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
    lineage_meta = {}
    bridge_events = []
    recent_history = {}
    prev_track_ids = set()
    seen_track_ids = set()

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
                'display_tracks': {},
            }

            if num_nodes == 0:
                gnn_processor.update(np.empty((0, 2)), None)
                frame_data['display_tracks'] = compute_display_tracks(gnn_processor, lineage_meta, bridge_events)
                update_recent_history(recent_history, frame_data['display_tracks'], t)
                prev_track_ids = set(frame_data['display_tracks'].keys())
                viz_frames.append(frame_data)
                print(f'Frame {t:02d} | Active Tracks: 0')
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

            point_track_ids, centers, _ = assign_tracks_to_points(
                pred_centers,
                pred_track_ids,
                det_centers,
                cluster_indices_list,
                num_nodes,
            )

            frame_data['track_ids'] = point_track_ids
            frame_data['centers'] = centers

            current_track_ids = set(gnn_processor.tracks.keys())
            for track_id in current_track_ids:
                lineage_meta.setdefault(track_id, {
                    'root_id': track_id,
                    'parents': tuple(),
                    'generation': 0,
                    'inherited_prefix': np.empty((0, 2), dtype=float),
                })

            frame_data['display_tracks'] = compute_display_tracks(gnn_processor, lineage_meta, bridge_events)
            new_track_ids = sorted(current_track_ids - seen_track_ids)
            lineage_events = create_lineage_events(new_track_ids, prev_track_ids, frame_data['display_tracks'], recent_history, t)
            apply_lineage_events(lineage_meta, bridge_events, lineage_events)
            frame_data['display_tracks'] = compute_display_tracks(gnn_processor, lineage_meta, bridge_events)

            seen_track_ids.update(current_track_ids)
            update_recent_history(recent_history, frame_data['display_tracks'], t)
            prev_track_ids = set(frame_data['display_tracks'].keys())

            viz_frames.append(frame_data)
            print(f"Frame {t:02d} | Active Tracks: {len(frame_data['centers'])}")

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
        display_tracks = data.get('display_tracks', {})

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

        for uid in sorted(display_tracks):
            track_info = display_tracks[uid]
            if track_info['age'] > VIZ_STYLE['ghost_age']:
                continue
            color = track_info['color']
            alpha_scale = 1.0 if track_info['age'] == 0 else VIZ_STYLE['ghost_alpha_scale']
            draw_fading_trail(ax, track_info['trace'], color, alpha_scale=alpha_scale)
            for bridge in track_info.get('bridges', []):
                draw_bridge(ax, bridge, color, frame_idx)

        unique_ids = [uid for uid in np.unique(track_ids) if uid != -1]
        for uid in unique_ids:
            mask = track_ids == uid
            track_info = display_tracks.get(uid, {})
            color = track_info.get('color', get_track_color(uid))
            raw_group_points = raw_pos[mask]
            group_points = plot_pos[mask]

            draw_group_region(ax, group_points, color)

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
            f'{labels["frame"]}: {frame_idx:02d}\n'
            f'{labels["active_groups"]}: {len(data["centers"])}'
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
        ax.set_title(labels['title'], fontsize=14, color=VIZ_STYLE['title_color'], pad=12, fontweight='semibold')
        ax.set_xlabel(labels['xlabel'], fontsize=10, color=VIZ_STYLE['tick_color'])
        ax.set_ylabel(labels['ylabel'], fontsize=10, color=VIZ_STYLE['tick_color'])

    os.makedirs(config.OUTPUT_GIF_DIR, exist_ok=True)
    path2 = os.path.join(config.OUTPUT_GIF_DIR, f'sim_data_{NUM}_track_result_paper.gif')
    ani = animation.FuncAnimation(fig, update, frames=len(viz_frames), interval=max(1, 1000 // VIZ_STYLE['fps']))
    ani.save(path2, writer='pillow', fps=VIZ_STYLE['fps'], dpi=VIZ_STYLE['dpi'])
    plt.close(fig)
    print(f'Saved {path2}')

    overview_path = save_track_overview(viz_frames, xlim, ylim, labels)
    if overview_path is not None:
        print(f'Saved {overview_path}')


if __name__ == '__main__':
    run_inference_and_viz()
