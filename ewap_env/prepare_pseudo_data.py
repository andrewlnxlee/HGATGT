import os
import sys
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def compute_pseudo_labels(obsmat_path):
    print(f"正在为 {obsmat_path} 生成物理规则伪标签...")
    data = np.loadtxt(obsmat_path)
    
    pair_counts = defaultdict(int)
    frames = np.unique(data[:, 0])
    
    for f in frames:
        peds = data[data[:, 0] == f]
        n = len(peds)
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = peds[i], peds[j]
                pid1, pid2 = int(p1[1]), int(p2[1])
                if pid1 > pid2: pid1, pid2 = pid2, pid1
                
                x1, y1 = p1[2], p1[4]
                x2, y2 = p2[2], p2[4]
                vx1, vy1 = p1[5], p1[7]
                vx2, vy2 = p2[5], p2[7]
                
                dist = np.hypot(x1 - x2, y1 - y2)
                v1_norm = np.hypot(vx1, vy1)
                v2_norm = np.hypot(vx2, vy2)
                
                if dist < 2.0: 
                    if v1_norm < 0.2 and v2_norm < 0.2:
                        pair_counts[(pid1, pid2)] += 1
                    elif v1_norm >= 0.2 and v2_norm >= 0.2:
                        cos_sim = (vx1*vx2 + vy1*vy2) / (v1_norm * v2_norm)
                        speed_diff = abs(v1_norm - v2_norm)
                        if cos_sim > 0.85 and speed_diff < 0.6:
                            pair_counts[(pid1, pid2)] += 1

    valid_pairs = [pair for pair, count in pair_counts.items() if count >= 5]
    print(f"[{obsmat_path}] 找到 {len(valid_pairs)} 对符合物理规律的同行者")
    
    all_pids = set(data[:, 1].astype(int))
    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}
    idx_to_pid = {i: pid for pid, i in pid_to_idx.items()}
    
    num_nodes = len(all_pids)
    if len(valid_pairs) > 0:
        row = [pid_to_idx[p[0]] for p in valid_pairs]
        col = [pid_to_idx[p[1]] for p in valid_pairs]
        data_ones = np.ones(len(valid_pairs))
        row_full = row + col
        col_full = col + row
        data_full = np.ones(len(row_full))
        adj = coo_matrix((data_full, (row_full, col_full)), shape=(num_nodes, num_nodes))
        n_comps, labels = connected_components(adj, directed=False)
    else:
        labels = np.arange(num_nodes)
        
    ped_to_group = {idx_to_pid[i]: labels[i] for i in range(num_nodes)}
    return ped_to_group

def generate_training_data(scene_name, ped_to_group, split='train'):
    obsmat_path = os.path.join("datasets/ewap_dataset", scene_name, "obsmat.txt")
    data = np.loadtxt(obsmat_path)
    
    frames_data = defaultdict(list)
    for row in data:
        fid = int(row[0])
        pid = int(row[1])
        x = row[2]
        y = row[4]
        frames_data[fid].append((pid, x, y))
        
    sorted_frames = sorted(frames_data.keys())
    all_frame_dicts = []
    
    for fid in sorted_frames:
        peds = frames_data[fid]
        if len(peds) == 0: continue
        
        meas_list = []
        label_list = []
        for pid, x, y in peds:
            # 同样使用 config 中的统一缩放
            scaled_x = x * config.COORD_SCALE + config.COORD_OFFSET[0]
            scaled_y = y * config.COORD_SCALE + config.COORD_OFFSET[1]
            meas_list.append([scaled_x, scaled_y])
            label_list.append(ped_to_group[pid])
            
        meas = np.array(meas_list)
        labels = np.array(label_list)
        
        gt_centers_list = []
        for gid in np.unique(labels):
            mask = labels == gid
            group_pts = meas[mask]
            gt_centers_list.append([gid, np.mean(group_pts[:, 0]), np.mean(group_pts[:, 1])])
            
        all_frame_dicts.append({
            'meas': meas,
            'labels': labels,
            'gt_centers': np.array(gt_centers_list)
        })
        
    output_dir = os.path.join(config.DATA_ROOT, f"finetune_{split}")
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_size = config.FRAMES_PER_SAMPLE
    episode_count = 0
    # 数据增强：使用滑动窗口来切分，步长设为一半，增加样本量
    step = chunk_size // 2
    for start in range(0, len(all_frame_dicts) - chunk_size + 1, step):
        episode = all_frame_dicts[start:start+chunk_size]
        save_path = os.path.join(output_dir, f"pseudo_{scene_name}_{episode_count:05d}.npy")
        np.save(save_path, episode, allow_pickle=True)
        episode_count += 1
        
    print(f"[{scene_name}] -> {split}: 生成了 {episode_count} 个微调 Episode 并保存至 {output_dir}")

if __name__ == "__main__":
    print("=" * 60)
    print("基于物理规则的真实数据集(伪标签)生成")
    print("=" * 60)
    
    # 仅使用 seq_eth
    for scene in ['seq_eth']:
        obsmat = f"datasets/ewap_dataset/{scene}/obsmat.txt"
        if os.path.exists(obsmat):
            ped_to_group = compute_pseudo_labels(obsmat)
            # 生成到 finetune_train
            generate_training_data(scene, ped_to_group, split='train')
            # 同时也生成一点到 finetune_val 验证损失
            generate_training_data(scene, ped_to_group, split='val')
    print("数据生成完毕。您现在可以运行 train.py 进行 Fine-tuning。")
