import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from tqdm import tqdm
import imageio
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

import config
from model import GNNGroupTracker
from trackers.gnn_processor import GNNPostProcessor


FPS_DT = 0.4
MODEL_CONN_RADIUS = 85.0
TRACK_ASSIGN_THRESH_PX = 8.0
MAX_RENDER_FRAMES = 1500
U_OFFSET = 50
V_OFFSET = -50


def parse_obsmat(filepath):
    data = np.loadtxt(filepath)
    frames = defaultdict(list)
    for row in data:
        frames[int(row[0])].append({
            'id': int(row[1]),
            'pos': [row[2], row[4]],
            'vel': [row[5], row[7]],
        })
    return frames


def world_points_to_pixel(points_world, H_inv, frame_height, u_offset=U_OFFSET, v_offset=V_OFFSET, flip_y=False):
    if len(points_world) == 0:
        return np.empty((0, 2), dtype=np.float32)

    homo = np.concatenate([
        points_world[:, 1:2],
        points_world[:, 0:1],
        np.ones((len(points_world), 1), dtype=np.float32),
    ], axis=1).T

    pixel = H_inv @ homo
    denom = pixel[2]
    valid = np.abs(denom) > 1e-8

    u = np.zeros(len(points_world), dtype=np.float32)
    v = np.zeros(len(points_world), dtype=np.float32)
    u[valid] = pixel[0, valid] / denom[valid]
    v[valid] = pixel[1, valid] / denom[valid]

    if flip_y:
        v = frame_height - 1 - v

    u += u_offset
    v += v_offset
    return np.stack([u, v], axis=1)


def scaled_to_world(points_scaled):
    return (points_scaled - np.array(config.COORD_OFFSET, dtype=np.float32)) / float(config.COORD_SCALE)


def build_group_detections(group_labels, corrected_pixels):
    unique_labels = [label for label in np.unique(group_labels) if label != -1]
    det_centers, det_shapes = [], []

    for label in unique_labels:
        idx = np.where(group_labels == label)[0]
        pts = corrected_pixels[idx]
        det_centers.append(np.mean(pts, axis=0))
        if len(pts) > 1:
            wh = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
        else:
            wh = np.array([6.0, 6.0], dtype=np.float32)
        det_shapes.append(np.maximum(wh, 4.0))

    if det_centers:
        det_centers = np.asarray(det_centers, dtype=np.float32)
        det_shapes = np.asarray(det_shapes, dtype=np.float32)
    else:
        det_centers = np.empty((0, 2), dtype=np.float32)
        det_shapes = np.empty((0, 2), dtype=np.float32)

    return unique_labels, det_centers, det_shapes


def visualize_hotel_on_video():
    scene_name = 'HOTEL'
    video_path = 'datasets/ewap_dataset/seq_hotel/seq_hotel.avi'
    h_matrix_path = 'datasets/ewap_dataset/seq_hotel/H.txt'
    obsmat_path = 'datasets/ewap_dataset/seq_hotel/obsmat.txt'
    model_path = config.EWAP_MODEL_USE_PATH
    save_path = os.path.join(config.OUTPUT_MP4_DIR, 'hotel_v2_closed_loop.mp4')
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    # Hotel 相对 ETH 需要单独做一次垂直镜像修正。
    flip_y = True

    H = np.loadtxt(h_matrix_path)
    H_inv = np.linalg.inv(H)
    obs_data = parse_obsmat(obsmat_path)
    sorted_frames = sorted(obs_data.keys())

    gnn_model = GNNGroupTracker().to(device)
    if os.path.exists(model_path):
        gnn_model.load_state_dict(torch.load(model_path, map_location=device))
        gnn_model.eval()

    tracker = GNNPostProcessor()
    rng = np.random.default_rng(2026)

    try:
        reader = imageio.get_reader(video_path)
    except Exception as e:
        print(f'❌ 无法读取视频: {e}')
        return

    writer = imageio.get_writer(save_path, fps=10, quality=9, codec='libx264')
    print('保存视频到: ', save_path)

    colors = {}
    track_history = defaultdict(list)

    stats_ari = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    stats_loop_error = []

    from prepare_pseudo_data import compute_pseudo_labels
    print('正在加载真值标签(物理规则基准)...')
    ped_to_group_gt = compute_pseudo_labels(obsmat_path)

    print(f'正在生成 {scene_name} 闭环校验可视化视频 (共 {len(sorted_frames[:MAX_RENDER_FRAMES])} 采样点)...')
    for fid in tqdm(sorted_frames[:MAX_RENDER_FRAMES]):
        try:
            frame = reader.get_data(fid)
        except:
            continue

        frame = frame.copy()
        frame_height = frame.shape[0]
        peds = obs_data[fid]
        if len(peds) == 0:
            writer.append_data(frame)
            continue

        raw_pos = np.asarray([p['pos'] for p in peds], dtype=np.float32)
        raw_vel = np.asarray([p['vel'] for p in peds], dtype=np.float32)

        # 这里必须与训练/数据集保持同尺度，否则模型输出会漂。
        scaled_pos = raw_pos * float(config.COORD_SCALE) + np.asarray(config.COORD_OFFSET, dtype=np.float32)
        scaled_vel = raw_vel * FPS_DT * float(config.COORD_SCALE)

        x_in_np = np.concatenate([scaled_pos, scaled_vel], axis=1)
        x_in = torch.tensor(x_in_np, dtype=torch.float32, device=device)

        dist_mat = cdist(scaled_pos, scaled_pos)
        src, dst = np.where((dist_mat < MODEL_CONN_RADIUS) & (dist_mat > 0))
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long, device=device)

        if len(src) > 0:
            pos_src, pos_dst = x_in[src, :2], x_in[dst, :2]
            vel_src, vel_dst = x_in[src, 2:], x_in[dst, 2:]
            rel_pos = pos_dst - pos_src
            dist = torch.norm(rel_pos, dim=1, keepdim=True)
            rel_v = vel_dst - vel_src
            v1_n = torch.norm(vel_src, dim=1, keepdim=True) + 1e-6
            v2_n = torch.norm(vel_dst, dim=1, keepdim=True) + 1e-6
            cos_sim = torch.sum(vel_src * vel_dst, dim=1, keepdim=True) / (v1_n * v2_n)
            edge_attr = torch.cat([rel_pos, dist, rel_v, cos_sim], dim=1)

            raw_vel_tensor = torch.tensor(raw_vel, dtype=torch.float32, device=device)
            raw_vel_src, raw_vel_dst = raw_vel_tensor[src], raw_vel_tensor[dst]
            raw_rel_v = raw_vel_dst - raw_vel_src
            raw_speed_diffs = torch.norm(raw_rel_v, dim=1)
            raw_v1_n = torch.norm(raw_vel_src, dim=1, keepdim=True) + 1e-6
            raw_v2_n = torch.norm(raw_vel_dst, dim=1, keepdim=True) + 1e-6
            raw_cos_sim = torch.sum(raw_vel_src * raw_vel_dst, dim=1, keepdim=True) / (raw_v1_n * raw_v2_n)
        else:
            edge_attr = torch.empty((0, 6), dtype=torch.float32, device=device)
            raw_speed_diffs = torch.empty((0,), dtype=torch.float32, device=device)
            raw_cos_sim = torch.empty((0, 1), dtype=torch.float32, device=device)

        graph = Data(x=x_in, edge_index=edge_index, edge_attr=edge_attr)
        with torch.no_grad():
            edge_scores, offsets, _, _ = gnn_model(graph)

        corrected_scaled = scaled_pos + offsets.detach().cpu().numpy()
        corrected_world = scaled_to_world(corrected_scaled)
        corrected_pixels = world_points_to_pixel(corrected_world, H_inv, frame_height, flip_y=flip_y)

        if len(edge_scores) > 0:
            max_score = edge_scores.max().item()
            dynamic_threshold = max(0.05, max_score * 0.28)
            prob_mask = edge_scores > dynamic_threshold
            motion_mask = (raw_cos_sim[:, 0] > 0.40) & (raw_speed_diffs < 0.7)
            mask = prob_mask & motion_mask
        else:
            mask = edge_scores > 0.5

        edges_np = edge_index[:, mask].detach().cpu().numpy()
        num_nodes = len(corrected_pixels)
        if edges_np.shape[1] > 0:
            adj = coo_matrix((np.ones(edges_np.shape[1]), (edges_np[0], edges_np[1])), shape=(num_nodes, num_nodes))
            _, group_labels = connected_components(adj, directed=False)
        else:
            group_labels = np.arange(num_nodes)

        gt_group_labels = np.array([ped_to_group_gt.get(p['id'], p['id'] + 1000) for p in peds])
        if len(gt_group_labels) > 1:
            ari = adjusted_rand_score(gt_group_labels, group_labels)
            stats_ari.append(ari)
            for i in range(len(peds)):
                for j in range(i + 1, len(peds)):
                    gt_same = gt_group_labels[i] == gt_group_labels[j]
                    pred_same = group_labels[i] == group_labels[j]
                    if gt_same and pred_same:
                        total_tp += 1
                    elif (not gt_same) and pred_same:
                        total_fp += 1
                    elif gt_same and (not pred_same):
                        total_fn += 1
                    else:
                        total_tn += 1

        unique_labels, det_centers_px, det_shapes_px = build_group_detections(group_labels, corrected_pixels)
        pred_centers_px, pred_ids, _ = tracker.update(det_centers_px, det_shapes_px)

        for center, tid in zip(pred_centers_px, pred_ids):
            tid = int(tid)
            track_history[tid].append(center.copy())
            if len(track_history[tid]) > 50:
                track_history[tid].pop(0)
            if tid not in colors:
                colors[tid] = tuple(int(x) for x in rng.integers(100, 255, size=3))

        label_to_tid = {}
        label_to_center = {}
        frame_loop_error = []
        if len(det_centers_px) > 0 and len(pred_centers_px) > 0:
            dist_map = cdist(det_centers_px, pred_centers_px)
            r_idx, c_idx = linear_sum_assignment(dist_map)
            for ri, ci in zip(r_idx, c_idx):
                if dist_map[ri, ci] < TRACK_ASSIGN_THRESH_PX:
                    label = unique_labels[ri]
                    label_to_tid[label] = int(pred_ids[ci])
                    label_to_center[label] = pred_centers_px[ci]
                    frame_loop_error.append(float(dist_map[ri, ci]))
                    stats_loop_error.append(float(dist_map[ri, ci]))

        overlay = frame.copy()
        for label in unique_labels:
            members = np.where(group_labels == label)[0]
            group_pts = np.round(corrected_pixels[members]).astype(np.int32)
            tid = label_to_tid.get(label)
            color = colors.get(tid, (180, 200, 255))

            if len(group_pts) > 1:
                if len(group_pts) == 2:
                    cv2.line(overlay, tuple(group_pts[0]), tuple(group_pts[1]), color, 28, lineType=cv2.LINE_AA)
                else:
                    hull = cv2.convexHull(group_pts)
                    cv2.fillConvexPoly(overlay, hull, color, lineType=cv2.LINE_AA)
                    cv2.polylines(overlay, [hull], True, color, 18, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

        for label in unique_labels:
            members = np.where(group_labels == label)[0]
            tid = label_to_tid.get(label)
            color = colors.get(tid, (180, 200, 255))
            pts = np.round(corrected_pixels[members]).astype(np.int32)

            for pt in pts:
                cv2.circle(frame, tuple(pt), 4, color, -1, lineType=cv2.LINE_AA)

            if tid is None:
                continue

            center = label_to_center.get(label, np.mean(corrected_pixels[members], axis=0))
            center_int = tuple(np.round(center).astype(np.int32))

            hist = np.array(track_history[tid][-25:], dtype=np.int32)
            for j in range(len(hist) - 1):
                alpha = (j + 1) / len(hist)
                trail_color = [int(c * alpha + 100 * (1 - alpha)) for c in color]
                cv2.line(frame, tuple(hist[j]), tuple(hist[j + 1]), trail_color, 2, lineType=cv2.LINE_AA)

            cv2.circle(frame, center_int, 6, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(tid), (center_int[0] + 8, center_int[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        mean_loop_err = float(np.mean(frame_loop_error)) if frame_loop_error else 0.0
        cv2.putText(frame, f'Frame {fid}', (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'LoopErr {mean_loop_err:.2f}px', (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.append_data(frame)

    reader.close()
    writer.close()

    avg_ari = np.mean(stats_ari) if stats_ari else 0.0
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    avg_loop_err = np.mean(stats_loop_error) if stats_loop_error else 0.0
    max_loop_err = np.max(stats_loop_error) if stats_loop_error else 0.0

    print('\n' + '=' * 50)
    print(f'📊 H-GAT 群体划分性能统计报告 ({scene_name} Dataset)')
    print('=' * 50)
    print(f'🔹 平均调整兰德系数 (ARI): {avg_ari:.4f}')
    print(f'🔹 两两聚类准确率 (Precision): {precision * 100:.2f}%')
    print(f'🔹 两两聚类召回率 (Recall): {recall * 100:.2f}%')
    print(f'🔹 综合 F1 分数 (F1-Score): {f1:.4f}')
    print(f'🔹 闭环平均误差 (Loop Error): {avg_loop_err:.2f}px')
    print(f'🔹 闭环最大误差 (Loop Error): {max_loop_err:.2f}px')
    print('=' * 50)


if __name__ == '__main__':
    visualize_hotel_on_video()
