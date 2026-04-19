import sys
import os
# 将父目录加入系统路径，以便正确导入 config, model 和 trackers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import DBSCAN
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

# --- 坐标转换与校准 ---
def world_to_pixel(x, y, H_inv, u_offset=50, v_offset=-50):
    point = np.array([y, x, 1.0]).reshape(3, 1) 
    pixel = H_inv @ point
    if pixel[2, 0] == 0: return 0, 0
    u = pixel[0, 0] / pixel[2, 0]
    v = pixel[1, 0] / pixel[2, 0]
    return int(u + u_offset), int(v + v_offset)

def parse_obsmat(filepath):
    data = np.loadtxt(filepath)
    frames = defaultdict(list)
    for row in data:
        # data: [frame, pid, px, pz, py, vx, vz, vy]
        frames[int(row[0])].append({
            'id': int(row[1]), 
            'pos': [row[2], row[4]],
            'vel': [row[5], row[7]]
        })
    return frames

def visualize_on_video():
    video_path = "datasets/ewap_dataset/seq_eth/seq_eth.avi"
    h_matrix_path = "datasets/ewap_dataset/seq_eth/H.txt"
    obsmat_path = "datasets/ewap_dataset/seq_eth/obsmat.txt"
    # 加载从头训练的模型
    model_path = config.EWAP_MODEL_USE_PATH
    save_path = os.path.join(config.OUTPUT_MP4_DIR, "eth_v1.mp4")
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    
    H = np.loadtxt(h_matrix_path)
    H_inv = np.linalg.inv(H)
    obs_data = parse_obsmat(obsmat_path)
    sorted_frames = sorted(obs_data.keys())

    gnn_model = GNNGroupTracker().to(device)
    if os.path.exists(model_path):
        gnn_model.load_state_dict(torch.load(model_path, map_location=device))
        gnn_model.eval()

    tracker = GNNPostProcessor()

    try:
        reader = imageio.get_reader(video_path)
    except Exception as e:
        print(f"❌ 无法读取视频: {e}")
        return

    
    # 使用 10fps，确保动作连贯
    writer = imageio.get_writer(save_path, fps=10, quality=9, codec='libx264')
    print("保存视频到: ", save_path)
    history = {} 
    colors = {} 
    
    # --- 统计容器 ---
    stats_ari = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    
    # 提前加载物理规则伪标签作为真值
    from prepare_pseudo_data import compute_pseudo_labels
    print("正在加载真值标签(物理规则基准)...")
    ped_to_group_gt = compute_pseudo_labels(obsmat_path)

    # 增加渲染长度到 1500 点
    print(f"正在生成终极版高清演示视频并统计准确率 (共 {len(sorted_frames[:1500])} 采样点)...")
    for fid in tqdm(sorted_frames[:1500]):
        try:
            frame = reader.get_data(fid)
        except: continue
        
        frame = frame.copy()
        peds = obs_data[fid]
        raw_pos = np.array([p['pos'] for p in peds])
        raw_vel = np.array([p['vel'] for p in peds])
        scaled_pos = raw_pos * config.COORD_SCALE + np.array(config.COORD_OFFSET)
        
        # --- H-GAT-GT 推断 (升级 4D 节点 + 6 维边) ---
        x_in_np = np.concatenate([scaled_pos, raw_vel], axis=1)
        x_in = torch.tensor(x_in_np, dtype=torch.float).to(device)
        
        dist_mat = cdist(scaled_pos, scaled_pos)
        # 黄金半径：85.0 (兼顾感知与精度)
        src, dst = np.where((dist_mat < 85.0) & (dist_mat > 0))
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long).to(device)
        
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
        else:
            edge_attr = torch.empty((0, 6), dtype=torch.float).to(device)

        graph = Data(x=x_in, edge_index=edge_index, edge_attr=edge_attr)
        with torch.no_grad():
            edge_scores, offsets, _, _ = gnn_model(graph)
            
        corrected = scaled_pos + offsets.cpu().numpy()
        
        # --- 聚类与后处理 (黄金平衡版：兼顾 Precision 与 Recall) ---
        if len(edge_scores) > 0:
            max_score = edge_scores.max().item()
            # 动态门槛调回 0.28，最小门槛 0.05
            dynamic_threshold = max(0.05, max_score * 0.28)
            prob_mask = edge_scores > dynamic_threshold
            
            cos_sims = edge_attr[:, 5]
            rel_vels = edge_attr[:, 3:5]
            speed_diffs = torch.norm(rel_vels, dim=1)
            
            # 重新加严运动约束
            motion_mask = (cos_sims > 0.40) & (speed_diffs < 0.7)
            mask = prob_mask & motion_mask
        else:
            mask = edge_scores > 0.5
            
        edges_np = edge_index[:, mask].cpu().numpy()
        num_nodes = len(scaled_pos)
        if edges_np.shape[1] > 0:
            adj = coo_matrix((np.ones(edges_np.shape[1]), (edges_np[0], edges_np[1])), shape=(num_nodes, num_nodes))
            n_comps, group_labels = connected_components(adj, directed=False)
        else:
            n_comps = num_nodes
            group_labels = np.arange(num_nodes)
        
        # --- 【新增】准确率统计逻辑 ---
        gt_group_labels = np.array([ped_to_group_gt.get(p['id'], p['id']+1000) for p in peds])
        if len(gt_group_labels) > 1:
            ari = adjusted_rand_score(gt_group_labels, group_labels)
            stats_ari.append(ari)
            for i in range(len(peds)):
                for j in range(i + 1, len(peds)):
                    gt_same = (gt_group_labels[i] == gt_group_labels[j])
                    pred_same = (group_labels[i] == group_labels[j])
                    if gt_same and pred_same: total_tp += 1
                    elif not gt_same and pred_same: total_fp += 1
                    elif gt_same and not pred_same: total_fn += 1
                    else: total_tn += 1

        # 1. 提取当前帧的聚类中心
        unique_labels = [l for l in np.unique(group_labels) if l != -1]
        det_c, det_s = [], []
        for l in unique_labels:
            idx = np.where(group_labels == l)[0]
            det_c.append(np.mean(corrected[idx], axis=0))
            pts_raw = scaled_pos[idx]
            wh = np.percentile(pts_raw, 95, axis=0) - np.percentile(pts_raw, 5, axis=0) if len(pts_raw)>1 else np.array([5,5])
            det_s.append(np.maximum(wh, 3.0))
        
        # 2. 更新跟踪器
        pred_c, pred_id, _ = tracker.update(np.array(det_c), np.array(det_s))

        # 3. 建立聚类标签到跟踪 ID 的映射
        label_to_tid = {}
        if len(det_c) > 0 and len(pred_c) > 0:
            dist_map = cdist(det_c, pred_c)
            r_idx, c_idx = linear_sum_assignment(dist_map)
            for ri, ci in zip(r_idx, c_idx):
                if dist_map[ri, ci] < 80:
                    label_to_tid[unique_labels[ri]] = pred_id[ci]

        # --- 渲染阶段 ---
        for l in unique_labels:
            members = np.where(group_labels == l)[0]
            if len(members) > 1:
                group_pts = []
                for midx in members:
                    u, v = world_to_pixel(raw_pos[midx][0], raw_pos[midx][1], H_inv)
                    group_pts.append([u, v])
                group_pts = np.array(group_pts, dtype=np.int32)
                overlay = frame.copy()
                if len(group_pts) == 2:
                    cv2.line(overlay, tuple(group_pts[0]), tuple(group_pts[1]), (180, 200, 255), 35, lineType=cv2.LINE_AA)
                else:
                    hull = cv2.convexHull(group_pts)
                    cv2.fillConvexPoly(overlay, hull, (180, 200, 255), lineType=cv2.LINE_AA)
                    cv2.polylines(overlay, [hull], True, (180, 200, 255), 35, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        for i in range(len(raw_pos)):
            l = group_labels[i]
            u, v = world_to_pixel(raw_pos[i][0], raw_pos[i][1], H_inv)
            if l in label_to_tid:
                tid = label_to_tid[l]
                if tid not in colors: colors[tid] = (np.random.randint(100,255), np.random.randint(100,255), np.random.randint(100,255))
                hist_key = f"{tid}_{i}"
                if hist_key not in history: history[hist_key] = []
                history[hist_key].append((u, v))
                pts = np.array(history[hist_key][-25:], dtype=np.int32)
                for j in range(len(pts)-1):
                    alpha = (j + 1) / len(pts)
                    thickness = int(1 + alpha * 3)
                    cv2.line(frame, tuple(pts[j]), tuple(pts[j+1]), colors[tid], thickness, lineType=cv2.LINE_AA)
                cv2.circle(frame, (u, v), 5, colors[tid], -1, lineType=cv2.LINE_AA)
                if len(pts) >= 2:
                    cv2.arrowedLine(frame, tuple(pts[-2]), tuple(pts[-1]), colors[tid], 2, tipLength=0.8, line_type=cv2.LINE_AA)
                cv2.putText(frame, str(tid), (u+5, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (u, v), 2, (150, 150, 150), -1, lineType=cv2.LINE_AA)

        writer.append_data(frame)

    reader.close()
    writer.close()
    
    # --- 输出报告 ---
    avg_ari = np.mean(stats_ari) if stats_ari else 0
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print("\n" + "="*50 + "\n📊 H-GAT 群体划分性能统计报告 (ETH Dataset)\n" + "="*50)
    print(f"🔹 平均调整兰德系数 (ARI): {avg_ari:.4f}\n🔹 两两聚类准确率 (Precision): {precision*100:.2f}%\n🔹 两两聚类召回率 (Recall): {recall*100:.2f}%\n🔹 综合 F1 分数 (F1-Score): {f1:.4f}\n" + "="*50)

if __name__ == "__main__":
    visualize_on_video()
