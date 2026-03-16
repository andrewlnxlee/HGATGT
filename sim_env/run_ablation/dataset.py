# dataset.py
import torch
import os
import numpy as np
from torch_geometric.data import Data, Dataset
from scipy.spatial.distance import cdist
import config

class RadarFileDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        self.root_dir = os.path.join(config.DATA_ROOT, split)
        if os.path.exists(self.root_dir):
            self.file_list = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.npy')])
        else:
            self.file_list = []
        self.conn_radius = 80.0 # 扩大建图半径以适应 ETH 尺度
        
    def len(self):
        return len(self.file_list)

    def get(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        episode_data = np.load(file_path, allow_pickle=True)
        
        graph_list = []
        
        for frame in episode_data:
            if frame is None or len(frame['meas']) == 0:
                continue
                
            meas = frame['meas'] # 可能 [N, 2] 或 [N, 4]
            labels = frame['labels']
            gt_centers = frame['gt_centers'] 
            
            # 1. 节点特征
            x = torch.tensor(meas, dtype=torch.float)
            num_nodes = x.size(0)
            
            # 2. 建图 (使用前2维位置进行距离判定)
            pos_only = meas[:, :2]
            dist_mat = cdist(pos_only, pos_only)
            src, dst = np.where((dist_mat < self.conn_radius) & (dist_mat > 0))
            edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
            
            # 3. 边属性 (根据 config.EDGE_DIM 自动适配)
            if len(src) > 0:
                pos_src = x[src, :2]
                pos_dst = x[dst, :2]
                rel_pos = pos_dst - pos_src
                dist = torch.norm(rel_pos, dim=1, keepdim=True)
                
                if x.size(1) >= 4: # 如果包含速度信息
                    # 显式取 [2, 3] 维，防止数据源有冗余列导致维度爆炸
                    v_src = x[src, 2:4]
                    v_dst = x[dst, 2:4]
                    rel_v = v_dst - v_src
                    v1_n = torch.norm(v_src, dim=1, keepdim=True) + 1e-6
                    v2_n = torch.norm(v_dst, dim=1, keepdim=True) + 1e-6
                    cos_sim = torch.sum(v_src * v_dst, dim=1, keepdim=True) / (v1_n * v2_n)
                    edge_attr = torch.cat([rel_pos, dist, rel_v, cos_sim], dim=1) # 严格 6 维 (2+1+2+1)
                else:
                    edge_attr = torch.cat([rel_pos, dist], dim=1) # 3维
            else:
                edge_attr = torch.empty((0, config.EDGE_DIM), dtype=torch.float)
            
            # 4. 边标签 (GT)
            l_src = labels[src]
            l_dst = labels[dst]
            edge_label = ((l_src == l_dst) & (l_src != 0)).astype(np.float32)
            edge_label = torch.tensor(edge_label)
            
            # 5. 封装
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                        edge_label=edge_label,
                        point_labels=torch.tensor(labels, dtype=torch.long),
                        gt_centers=torch.tensor(gt_centers, dtype=torch.float))
            graph_list.append(data)
            
        return graph_list
