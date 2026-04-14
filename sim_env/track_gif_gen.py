import os
import sys
from collections import defaultdict

# 添加根目录路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError
from sklearn.metrics import accuracy_score

import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from trackers.gnn_processor_all import (
    GROUP_TO_CLUSTER_THRESH,
    HierarchicalTrackProcessor,
    assign_track_ids_to_points,
)


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
    'continuous_width': 3.2,
    'continuous_alpha': 0.92,
    'continuous_point_size': 16,
    'continuous_lighten_factor': 0.72,
    'merge_link_alpha': 0.82,
    'merge_link_width': 2.6,
    'merge_marker_size': 62,
    'centroid_outer_size': 132,
    'centroid_inner_size': 58,
    'label_fontsize': 9,
    'info_fontsize': 10,
    'margin_ratio': 0.08,
    'relation_frames': 6,
    'relation_min_support_points': 3,
    'relation_min_parent_ratio': 0.25,
    'relation_min_child_ratio': 0.25,
    'relation_arrow_width': 2.8,
    'relation_arrow_head_scale': 22,
    'relation_arrow_outline_width': 4.6,
    'relation_merge_color': '#D84B3A',
    'relation_split_color': '#2D8C82',
    'relation_birth_color': '#7B8794',
    'relation_alpha': 0.88,
    'relation_label_fontsize': 8.5,
    'relation_marker_outer_size': 94,
    'relation_marker_inner_size': 48,
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


def build_center_lookup(group_ids, group_centers):
    group_ids = np.asarray(group_ids, dtype=int).reshape(-1)
    group_centers = np.asarray(group_centers, dtype=float).reshape(-1, 2)
    return {int(gid): np.asarray(center, dtype=float) for gid, center in zip(group_ids, group_centers)}


def format_group_ids(group_ids):
    return ' + '.join(f'G{int(group_id)}' for group_id in group_ids)


def compute_display_tracks(group_tracker):
    display_tracks = {}
    for track_id, trk in group_tracker.tracks.items():
        color = get_track_color(track_id)
        display_tracks[track_id] = {
            'trace': np.asarray(trk.get('trace', []), dtype=float),
            'age': int(trk.get('age', 0)),
            'is_observed': int(trk.get('age', 0)) == 0,
            'center': np.asarray(trk.get('last_meas', trk['x'][:2]), dtype=float),
            'pred_center': np.asarray(trk['x'][:2], dtype=float),
            'color': color,
            'shape': np.asarray(trk.get('shape', np.array([3.0, 3.0])), dtype=float),
        }
    return display_tracks


def compute_raw_tracks(group_tracker):
    raw_tracks = {}
    for track_id, trk in group_tracker.tracks.items():
        raw_tracks[track_id] = {
            'trace': np.asarray(trk.get('trace', []), dtype=float),
            'color': get_track_color(track_id),
            'age': int(trk.get('age', 0)),
        }
    return raw_tracks


def _relation_pair_is_valid(count, parent_total, child_total):
    if count < VIZ_STYLE['relation_min_support_points']:
        return False
    if parent_total <= 0 or child_total <= 0:
        return False
    parent_ratio = count / parent_total
    child_ratio = count / child_total
    return (
        parent_ratio >= VIZ_STYLE['relation_min_parent_ratio']
        and child_ratio >= VIZ_STYLE['relation_min_child_ratio']
    )


def infer_frame_relations(prev_frame, curr_frame, frame_idx):
    curr_point_group_ids = np.asarray(curr_frame.get('point_group_ids', []), dtype=int).reshape(-1)
    curr_group_ids = np.asarray(curr_frame.get('group_ids', []), dtype=int).reshape(-1)
    curr_group_centers = np.asarray(curr_frame.get('group_centers', np.empty((0, 2))), dtype=float).reshape(-1, 2)
    curr_detected_centers = np.asarray(curr_frame.get('detected_centers', np.empty((0, 2))), dtype=float).reshape(-1, 2)
    curr_centroid_to_points = curr_frame.get('centroid_to_points', {})
    curr_num_points = len(curr_frame.get('plot_pos', np.empty((0, 2))))

    curr_center_lookup = build_center_lookup(curr_group_ids, curr_group_centers)
    curr_frame['prev_point_group_ids'] = np.full(curr_num_points, -1, dtype=int)

    if prev_frame is None:
        relations = []
        for child_id in sorted(curr_group_ids):
            child_center = curr_center_lookup.get(int(child_id))
            relations.append({
                'type': 'birth',
                'frame_idx': frame_idx,
                'parents': tuple(),
                'children': (int(child_id),),
                'support_count': 0,
                'support_details': tuple(),
                'parent_centers': [],
                'child_centers': [child_center] if child_center is not None else [],
                'anchor': child_center,
                'label': f'G{int(child_id)}',
            })
        return relations

    prev_group_ids = np.asarray(prev_frame.get('group_ids', []), dtype=int).reshape(-1)
    prev_group_centers = np.asarray(prev_frame.get('group_centers', np.empty((0, 2))), dtype=float).reshape(-1, 2)
    prev_center_lookup = build_center_lookup(prev_group_ids, prev_group_centers)

    prev_point_group_ids = assign_track_ids_to_points(
        prev_group_centers,
        prev_group_ids,
        curr_detected_centers,
        curr_centroid_to_points,
        curr_num_points,
        GROUP_TO_CLUSTER_THRESH,
    )
    curr_frame['prev_point_group_ids'] = prev_point_group_ids

    pair_counts = defaultdict(int)
    parent_totals = defaultdict(int)
    child_totals = defaultdict(int)

    for parent_id in prev_point_group_ids[prev_point_group_ids >= 0]:
        parent_totals[int(parent_id)] += 1
    for child_id in curr_point_group_ids[curr_point_group_ids >= 0]:
        child_totals[int(child_id)] += 1

    valid_mask = (prev_point_group_ids >= 0) & (curr_point_group_ids >= 0)
    for parent_id, child_id in zip(prev_point_group_ids[valid_mask], curr_point_group_ids[valid_mask]):
        pair_counts[(int(parent_id), int(child_id))] += 1

    child_supports = defaultdict(list)
    parent_supports = defaultdict(list)
    for (parent_id, child_id), count in sorted(pair_counts.items()):
        parent_total = parent_totals[parent_id]
        child_total = child_totals[child_id]
        parent_ratio = count / parent_total if parent_total > 0 else 0.0
        child_ratio = count / child_total if child_total > 0 else 0.0
        if not _relation_pair_is_valid(count, parent_total, child_total):
            continue
        support = {
            'parent_id': parent_id,
            'child_id': child_id,
            'count': int(count),
            'parent_ratio': float(parent_ratio),
            'child_ratio': float(child_ratio),
        }
        child_supports[child_id].append(support)
        parent_supports[parent_id].append(support)

    for child_id in child_supports:
        child_supports[child_id].sort(key=lambda item: (-item['count'], -item['parent_ratio'], item['parent_id']))
    for parent_id in parent_supports:
        parent_supports[parent_id].sort(key=lambda item: (-item['count'], -item['child_ratio'], item['child_id']))

    relations = []
    merge_children = {child_id for child_id, supports in child_supports.items() if len(supports) >= 2}
    split_parents = {parent_id for parent_id, supports in parent_supports.items() if len(supports) >= 2}

    for child_id in sorted(merge_children):
        supports = child_supports[child_id]
        parents = tuple(int(item['parent_id']) for item in supports)
        child_center = curr_center_lookup.get(int(child_id))
        parent_centers = [prev_center_lookup[parent_id] for parent_id in parents if parent_id in prev_center_lookup]
        relations.append({
            'type': 'merge',
            'frame_idx': frame_idx,
            'parents': parents,
            'children': (int(child_id),),
            'support_count': int(sum(item['count'] for item in supports)),
            'support_details': tuple(supports),
            'parent_centers': parent_centers,
            'child_centers': [child_center] if child_center is not None else [],
            'anchor': child_center,
            'label': f'{format_group_ids(parents)} -> G{int(child_id)}',
        })

    for parent_id in sorted(split_parents):
        supports = parent_supports[parent_id]
        children = tuple(int(item['child_id']) for item in supports)
        parent_center = prev_center_lookup.get(int(parent_id))
        child_centers = [curr_center_lookup[child_id] for child_id in children if child_id in curr_center_lookup]
        relations.append({
            'type': 'split',
            'frame_idx': frame_idx,
            'parents': (int(parent_id),),
            'children': children,
            'support_count': int(sum(item['count'] for item in supports)),
            'support_details': tuple(supports),
            'parent_centers': [parent_center] if parent_center is not None else [],
            'child_centers': child_centers,
            'anchor': parent_center,
            'label': f'G{int(parent_id)} -> {format_group_ids(children)}',
        })

    used_pairs = set()
    for child_id, supports in child_supports.items():
        if child_id in merge_children:
            used_pairs.update((item['parent_id'], item['child_id']) for item in supports)
            continue
        if len(supports) != 1:
            continue
        support = supports[0]
        parent_id = int(support['parent_id'])
        if parent_id in split_parents:
            used_pairs.add((parent_id, int(child_id)))
            continue
        child_center = curr_center_lookup.get(int(child_id))
        parent_center = prev_center_lookup.get(parent_id)
        relations.append({
            'type': 'continue',
            'frame_idx': frame_idx,
            'parents': (parent_id,),
            'children': (int(child_id),),
            'support_count': int(support['count']),
            'support_details': (support,),
            'parent_centers': [parent_center] if parent_center is not None else [],
            'child_centers': [child_center] if child_center is not None else [],
            'anchor': child_center if child_center is not None else parent_center,
            'label': f'G{parent_id} -> G{int(child_id)}',
        })
        used_pairs.add((parent_id, int(child_id)))

    supported_children = set(child_supports.keys())
    supported_parents = set(parent_supports.keys())

    for child_id in sorted(int(group_id) for group_id in curr_group_ids if int(group_id) not in supported_children):
        child_center = curr_center_lookup.get(child_id)
        relations.append({
            'type': 'birth',
            'frame_idx': frame_idx,
            'parents': tuple(),
            'children': (child_id,),
            'support_count': 0,
            'support_details': tuple(),
            'parent_centers': [],
            'child_centers': [child_center] if child_center is not None else [],
            'anchor': child_center,
            'label': f'G{child_id}',
        })

    for parent_id in sorted(int(group_id) for group_id in prev_group_ids if int(group_id) not in supported_parents):
        parent_center = prev_center_lookup.get(parent_id)
        relations.append({
            'type': 'death',
            'frame_idx': frame_idx,
            'parents': (parent_id,),
            'children': tuple(),
            'support_count': 0,
            'support_details': tuple(),
            'parent_centers': [parent_center] if parent_center is not None else [],
            'child_centers': [],
            'anchor': parent_center,
            'label': f'G{parent_id}',
        })

    relations.sort(key=lambda item: ({'merge': 0, 'split': 1, 'continue': 2, 'birth': 3, 'death': 4}[item['type']], item['label']))
    return relations


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
            trace = np.asarray(track_info.get('trace', []), dtype=float)
            if trace.size:
                collected.append(trace)
        for relation in data.get('relations', []):
            relation_points = []
            relation_points.extend(relation.get('parent_centers', []))
            relation_points.extend(relation.get('child_centers', []))
            if relation_points:
                collected.append(np.vstack(relation_points))

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
    trace = np.asarray(trace, dtype=float)
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


def _draw_relation_arrow(ax, start, end, color, alpha, zorder):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if np.linalg.norm(end - start) < 1e-6:
        return

    ax.annotate(
        '',
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle='-|>',
            color=mcolors.to_rgba('white', min(alpha + 0.08, 0.98)),
            lw=VIZ_STYLE['relation_arrow_outline_width'],
            shrinkA=8,
            shrinkB=8,
            mutation_scale=VIZ_STYLE['relation_arrow_head_scale'],
        ),
        zorder=zorder,
    )
    ax.annotate(
        '',
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle='-|>',
            color=mcolors.to_rgba(color, alpha),
            lw=VIZ_STYLE['relation_arrow_width'],
            shrinkA=8,
            shrinkB=8,
            mutation_scale=VIZ_STYLE['relation_arrow_head_scale'],
        ),
        zorder=zorder + 0.1,
    )


def _relation_color(relation_type):
    if relation_type == 'merge':
        return VIZ_STYLE['relation_merge_color']
    if relation_type == 'split':
        return VIZ_STYLE['relation_split_color']
    return VIZ_STYLE['relation_birth_color']


def draw_relation_event(ax, relation, current_frame_idx=None, overview=False):
    relation_type = relation.get('type')
    if relation_type not in {'merge', 'split'}:
        return

    if overview:
        fade = 1.0
    else:
        age = current_frame_idx - int(relation['frame_idx'])
        if age < 0 or age >= VIZ_STYLE['relation_frames']:
            return
        fade = 1.0 - age / max(VIZ_STYLE['relation_frames'], 1)

    color = _relation_color(relation_type)
    alpha = VIZ_STYLE['relation_alpha'] * fade

    if relation_type == 'merge':
        child_centers = relation.get('child_centers', [])
        if not child_centers:
            return
        child_center = np.asarray(child_centers[0], dtype=float)
        for parent_center in relation.get('parent_centers', []):
            _draw_relation_arrow(ax, parent_center, child_center, color, alpha, zorder=8)

        ax.scatter(
            child_center[0], child_center[1],
            s=VIZ_STYLE['relation_marker_outer_size'],
            color=mcolors.to_rgba('white', min(alpha + 0.04, 0.98)),
            edgecolors='none',
            zorder=9,
        )
        ax.scatter(
            child_center[0], child_center[1],
            s=VIZ_STYLE['relation_marker_inner_size'],
            color=mcolors.to_rgba(color, alpha),
            marker='P',
            edgecolors='white',
            linewidths=0.8,
            zorder=10,
        )
        anchor = child_center
    else:
        parent_centers = relation.get('parent_centers', [])
        if not parent_centers:
            return
        parent_center = np.asarray(parent_centers[0], dtype=float)
        for child_center in relation.get('child_centers', []):
            _draw_relation_arrow(ax, parent_center, child_center, color, alpha, zorder=8)

        ax.scatter(
            parent_center[0], parent_center[1],
            s=VIZ_STYLE['relation_marker_outer_size'],
            color=mcolors.to_rgba('white', min(alpha + 0.04, 0.98)),
            edgecolors='none',
            zorder=9,
        )
        ax.scatter(
            parent_center[0], parent_center[1],
            s=VIZ_STYLE['relation_marker_inner_size'],
            color=mcolors.to_rgba(color, alpha),
            marker='X',
            edgecolors='white',
            linewidths=0.8,
            zorder=10,
        )
        anchor = parent_center

    offset = (10, 9) if (int(relation['frame_idx']) + (0 if relation_type == 'merge' else 1)) % 2 else (10, -16)
    va = 'bottom' if offset[1] > 0 else 'top'
    label = relation['label']
    if overview:
        label = f'{label}\n(t={int(relation["frame_idx"]):02d})'

    ax.annotate(
        label,
        xy=(anchor[0], anchor[1]),
        xytext=offset,
        textcoords='offset points',
        color=color,
        fontsize=VIZ_STYLE['relation_label_fontsize'],
        fontweight='semibold',
        ha='left',
        va=va,
        bbox=dict(
            boxstyle='round,pad=0.24',
            fc=mcolors.to_rgba('white', 0.90),
            ec=mcolors.to_rgba(color, min(alpha, 0.65)),
            lw=0.85,
        ),
        zorder=11,
    )


def collect_overview_tracks(viz_frames, field_name):
    overview_tracks = {}

    for data in viz_frames:
        for track_id, track_info in data.get(field_name, {}).items():
            trace = np.asarray(track_info.get('trace', []), dtype=float)
            if not trace.size:
                continue
            existing = overview_tracks.get(track_id)
            if existing is None or len(trace) >= len(existing['trace']):
                overview_tracks[track_id] = {
                    'trace': trace.copy(),
                    'color': track_info['color'],
                }

    return overview_tracks


def save_track_overview(viz_frames, xlim, ylim, labels, relation_events):
    raw_overview_tracks = collect_overview_tracks(viz_frames, 'raw_tracks')
    continuous_overview_tracks = collect_overview_tracks(viz_frames, 'display_tracks')
    if not raw_overview_tracks and not continuous_overview_tracks:
        return None

    fig, ax = plt.subplots(figsize=VIZ_STYLE['figsize'], dpi=VIZ_STYLE['dpi'])
    fig.patch.set_facecolor(VIZ_STYLE['background'])
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=0.97)
    style_axes(ax)

    for track_id in sorted(raw_overview_tracks):
        track_info = raw_overview_tracks[track_id]
        trace = track_info['trace']
        color = track_info['color']
        ax.plot(
            trace[:, 0], trace[:, 1],
            color=mcolors.to_rgba(color, 0.26),
            linewidth=1.25,
            solid_capstyle='round',
            zorder=1,
        )
        ax.scatter(
            trace[:, 0], trace[:, 1],
            color=mcolors.to_rgba(color, 0.34),
            s=9,
            edgecolors='none',
            zorder=2,
        )

    for track_id in sorted(continuous_overview_tracks):
        track_info = continuous_overview_tracks[track_id]
        trace = track_info['trace']
        color = adjust_color(track_info['color'], VIZ_STYLE['continuous_lighten_factor'])
        ax.plot(
            trace[:, 0], trace[:, 1],
            color=mcolors.to_rgba('white', 0.98),
            linewidth=VIZ_STYLE['continuous_width'] + 1.6,
            solid_capstyle='round',
            zorder=3,
        )
        ax.plot(
            trace[:, 0], trace[:, 1],
            color=mcolors.to_rgba(color, VIZ_STYLE['continuous_alpha']),
            linewidth=VIZ_STYLE['continuous_width'],
            solid_capstyle='round',
            zorder=4,
        )
        ax.scatter(
            trace[:, 0], trace[:, 1],
            color=mcolors.to_rgba(color, 0.88),
            s=VIZ_STYLE['continuous_point_size'],
            edgecolors='white',
            linewidths=0.25,
            zorder=5,
        )
        ax.scatter(
            trace[0, 0], trace[0, 1],
            color=mcolors.to_rgba(color, 0.82),
            s=28,
            marker='o',
            edgecolors='white',
            linewidths=0.7,
            zorder=6,
        )
        ax.scatter(
            trace[-1, 0], trace[-1, 1],
            color=color,
            s=40,
            marker='X',
            edgecolors='white',
            linewidths=0.7,
            zorder=7,
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
            zorder=8,
        )

    for relation in relation_events:
        rel_type = relation.get('type')
        if rel_type in {'merge', 'split'}:
            parents = relation.get('parents', [])
            children = relation.get('children', [])
            
            if rel_type == 'merge' and children:
                cid = children[0]
                if cid in continuous_overview_tracks:
                    c_pt = continuous_overview_tracks[cid]['trace'][0]
                    for pid in parents:
                        if pid in continuous_overview_tracks:
                            p_pt = continuous_overview_tracks[pid]['trace'][-1]
                            p_color = continuous_overview_tracks[pid]['color']
                            ax.plot(
                                [p_pt[0], c_pt[0]], [p_pt[1], c_pt[1]],
                                color=mcolors.to_rgba(p_color, VIZ_STYLE['merge_link_alpha']),
                                linestyle='--',
                                linewidth=VIZ_STYLE['merge_link_width'],
                                zorder=4.5
                            )
            elif rel_type == 'split' and parents:
                pid = parents[0]
                if pid in continuous_overview_tracks:
                    p_pt = continuous_overview_tracks[pid]['trace'][-1]
                    for cid in children:
                        if cid in continuous_overview_tracks:
                            c_pt = continuous_overview_tracks[cid]['trace'][0]
                            c_color = continuous_overview_tracks[cid]['color']
                            ax.plot(
                                [p_pt[0], c_pt[0]], [p_pt[1], c_pt[1]],
                                color=mcolors.to_rgba(c_color, VIZ_STYLE['merge_link_alpha']),
                                linestyle='--',
                                linewidth=VIZ_STYLE['merge_link_width'],
                                zorder=4.5
                            )

        draw_relation_event(ax, relation, overview=True)

    info_text = (
        f'{labels["frame"]}: 00-{len(viz_frames) - 1:02d}\n'
        f'{labels["active_groups"]}: {len(continuous_overview_tracks) or len(raw_overview_tracks)}'
    )
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=VIZ_STYLE['info_fontsize'],
        color=VIZ_STYLE['text_color'],
        bbox=dict(boxstyle='round,pad=0.35', fc=mcolors.to_rgba('white', 0.80), ec='none'),
        zorder=12,
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
# 推理主流程（对齐 evaluate_all.py 的 group tracking 标签维系）
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

    track_processor = HierarchicalTrackProcessor()
    prev_frame_state = None
    viz_frames = []
    all_relation_events = []

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
                'point_group_ids': np.full(num_nodes, -1, dtype=int),
                'prev_point_group_ids': np.full(num_nodes, -1, dtype=int),
                'group_ids': np.zeros((0,), dtype=int),
                'group_centers': np.zeros((0, 2), dtype=float),
                'detected_centers': np.zeros((0, 2), dtype=float),
                'detected_shapes': None,
                'centroid_to_points': {},
                'centers': {},
                'acc': 0.0,
                'display_tracks': {},
                'raw_tracks': {},
                'relations': [],
            }

            if num_nodes == 0:
                group_out = track_processor.update_group_tracks(np.empty((0, 2)))
                frame_data['group_ids'] = group_out['group_ids']
                frame_data['group_centers'] = group_out['group_centers']
                frame_data['detected_centers'] = group_out['detected_centers']
                frame_data['detected_shapes'] = group_out['detected_shapes']
                frame_data['centroid_to_points'] = group_out['centroid_to_points']
                frame_data['point_group_ids'] = group_out['point_group_ids']
                frame_data['track_ids'] = group_out['point_group_ids']
                frame_data['centers'] = build_center_lookup(group_out['group_ids'], group_out['group_centers'])
                frame_data['display_tracks'] = compute_display_tracks(track_processor.group_tracker)
                frame_data['raw_tracks'] = compute_raw_tracks(track_processor.group_tracker)
                frame_data['relations'] = infer_frame_relations(prev_frame_state, frame_data, t)
                all_relation_events.extend(relation for relation in frame_data['relations'] if relation['type'] in {'merge', 'split'})
                prev_frame_state = frame_data
                viz_frames.append(frame_data)
                print(f'Frame {t:02d} | Active Tracks: 0')
                continue

            model_out = model(graph_dev)
            edge_scores = unpack_edge_scores(model_out)
            group_offsets = unpack_group_offsets(model_out)
            frame_data['acc'] = compute_edge_accuracy(edge_scores, graph_dev)

            corrected_pos = raw_pos + group_offsets.detach().cpu().numpy()
            frame_data['plot_pos'] = corrected_pos

            group_out = track_processor.update_group_tracks(corrected_pos)
            frame_data['group_ids'] = group_out['group_ids']
            frame_data['group_centers'] = group_out['group_centers']
            frame_data['detected_centers'] = group_out['detected_centers']
            frame_data['detected_shapes'] = group_out['detected_shapes']
            frame_data['centroid_to_points'] = group_out['centroid_to_points']
            frame_data['point_group_ids'] = group_out['point_group_ids']
            frame_data['track_ids'] = group_out['point_group_ids']
            frame_data['centers'] = build_center_lookup(group_out['group_ids'], group_out['group_centers'])
            frame_data['display_tracks'] = compute_display_tracks(track_processor.group_tracker)
            frame_data['raw_tracks'] = compute_raw_tracks(track_processor.group_tracker)
            frame_data['relations'] = infer_frame_relations(prev_frame_state, frame_data, t)

            all_relation_events.extend(relation for relation in frame_data['relations'] if relation['type'] in {'merge', 'split'})
            prev_frame_state = frame_data
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

        relation_start = max(0, frame_idx - VIZ_STYLE['relation_frames'] + 1)
        for relation_frame_idx in range(relation_start, frame_idx + 1):
            for relation in viz_frames[relation_frame_idx].get('relations', []):
                draw_relation_event(ax, relation, current_frame_idx=frame_idx, overview=False)

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
                    zorder=12,
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
            zorder=13,
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

    overview_path = save_track_overview(viz_frames, xlim, ylim, labels, all_relation_events)
    if overview_path is not None:
        print(f'Saved {overview_path}')


if __name__ == '__main__':
    run_inference_and_viz()
