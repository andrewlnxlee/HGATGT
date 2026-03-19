# dataset.py
import torch
import os
import numpy as np
from torch_geometric.data import Data, Dataset
from scipy.spatial.distance import cdist
import config

class RadarFileDataset(Dataset):
    def __init__(self, split='train', include_empty=False):
        super().__init__()
        self.root_dir = os.path.join(config.DATA_ROOT, split)
        self.file_list = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.npy')])
        self.conn_radius = 30.0
        self.include_empty = include_empty
        
    def len(self):
        # 实际上我们这里返回的是 (Samples * Frames) 的总帧数，
        # 或者是 Samples 数。为了简单训练，我们把每个 episode 视为一个 batch 里的列表
        # 这里为了配合 PyG Loader，我们把每一帧都当作单独的图来训练
        # 注意：这里需要预先扫描总帧数，为了演示方便，我们简化为：
        # 一个 Dataset item = 一个 Episode (包含 50 帧)
        return len(self.file_list)

    def get(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        episode_data = np.load(file_path, allow_pickle=True)
        
        graph_list = []
        
        # 将一个 Episode 里的每一帧都转成图
        for frame in episode_data:
            if frame is None:
                continue

            meas = np.asarray(frame['meas'], dtype=np.float32).reshape(-1, 2)
            if len(meas) == 0 and not self.include_empty:
                continue

            labels = np.asarray(frame['labels'], dtype=np.int64).reshape(-1)
            gt_centers = np.asarray(frame['gt_centers'], dtype=np.float32).reshape(-1, 3)
            has_gt_points = 'gt_points' in frame
            has_point_ids = 'point_ids' in frame
            gt_points = np.asarray(frame.get('gt_points', np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
            point_ids = np.asarray(frame.get('point_ids', np.zeros((len(meas),), dtype=np.int64)), dtype=np.int64).reshape(-1)

            # 1. 节点特征
            x = torch.tensor(meas, dtype=torch.float)

            # 2. 建图 (KNN / Radius)
            if len(meas) > 0:
                dist_mat = cdist(meas, meas)
                src, dst = np.where((dist_mat < self.conn_radius) & (dist_mat > 0))
                edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
                pos_src = x[src]
                pos_dst = x[dst]
                rel_pos = pos_dst - pos_src
                dist = torch.norm(rel_pos, dim=1, keepdim=True)
                edge_attr = torch.cat([rel_pos, dist], dim=1)
                l_src = labels[src]
                l_dst = labels[dst]
                edge_label = ((l_src == l_dst) & (l_src != 0)).astype(np.float32)
                edge_label = torch.tensor(edge_label)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 3), dtype=torch.float)
                edge_label = torch.zeros((0,), dtype=torch.float)

            # 5. 封装
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_label=edge_label,
                point_labels=torch.tensor(labels, dtype=torch.long),
                gt_centers=torch.tensor(gt_centers, dtype=torch.float),
                gt_points=torch.tensor(gt_points, dtype=torch.float),
                point_ids=torch.tensor(point_ids, dtype=torch.long)
            )
            data.has_gt_points = has_gt_points
            data.has_point_ids = has_point_ids
            graph_list.append(data)
            
        return graph_list # 返回一个列表，代表这一段时序数据