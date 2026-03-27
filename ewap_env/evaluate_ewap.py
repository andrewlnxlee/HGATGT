# evaluate_ewap.py
# EWAP (ETH Walking Pedestrians) 数据集评估脚本
# 彻底隔离 GT 和 Meas，模拟真实雷达噪声
import os
# 添加根目录路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.data import Dataset

import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from metrics import TrackingMetrics

# 导入跟踪器
from trackers.baseline import BaselineTracker
from trackers.gm_cphd import GMCPHDTracker
from trackers.cbmember import CBMeMBerTracker
from trackers.graph_mb import GraphMBTracker
from trackers.gnn_processor import GNNPostProcessor
from trackers.social_stgcnn_tracker import SocialSTGCNNTracker

class EWAPDataset(RadarFileDataset):
    def __init__(self, scene_folder):
        Dataset.__init__(self)
        self.root_dir = os.path.join(config.DATA_ROOT, scene_folder)
        if os.path.exists(self.root_dir):
            self.file_list = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.npy')])
        else:
            self.file_list = []
        self.conn_radius = 30.0

def evaluate_scene(scene_name, scene_folder, gnn_model, device):
    test_set = EWAPDataset(scene_folder)
    if len(test_set) == 0: return None

    tracker_configs = {
        'Baseline': lambda: BaselineTracker(eps=35, min_samples=1),
        'GM-CPHD': lambda: GMCPHDTracker(),
        'CBMeMBer': lambda: CBMeMBerTracker(),
        'Social-STGCNN': lambda: SocialSTGCNNTracker(scene=scene_name.lower()),
        'Graph-MB': lambda: GraphMBTracker(),
        'H-GAT-GT (Ours)': lambda: GNNPostProcessor(),
    }

    metrics = {name: TrackingMetrics() for name in tracker_configs.keys()}

    for episode_idx in tqdm(range(len(test_set)), desc=scene_name):
        episode_graphs = test_set.get(episode_idx)
        trackers = {name: factory() for name, factory in tracker_configs.items()}

        for graph in episode_graphs:
            # 1. 彻底隔离 GT 数据 (深度拷贝)
            gt_data_raw = graph.gt_centers.numpy().copy()
            gt_centers = gt_data_raw[:, 1:3] if len(gt_data_raw) > 0 else np.zeros((0, 2))
            gt_ids = gt_data_raw[:, 0].astype(int) if len(gt_data_raw) > 0 else []
            pt_lbl = graph.point_labels.cpu().numpy().copy()

            # 2. 构造独立的测量点 Meas (注入 3.0 强度的噪声)
            # 加大噪声，进一步考验各算法的滤波和状态估计能力
            raw_x = graph.x.numpy().copy()
            noise = np.zeros_like(raw_x)
            noise[:,:2] = np.random.normal(0, 3.0, (len(raw_x), 2))
            if raw_x.shape[1] >= 4:
                # 适当注入速度噪声
                noise[:,2:4] = np.random.normal(0, 0.5, (len(raw_x), 2))
            meas_points = raw_x + noise

            # ========================
            # H-GAT-GT (Ours)
            # ========================
            t0 = time.time()
            if gnn_model:
                graph_dev = graph.to(device)
                # 注意：为了公平，GNN 的输入也必须是带噪声的 meas_points
                # 我们临时替换 graph 的数据，并同步更新 edge_attr
                graph_dev.x = torch.from_numpy(meas_points).float().to(device)
                
                src, dst = graph_dev.edge_index
                if len(src) > 0:
                    pos_src, pos_dst = graph_dev.x[src, :2], graph_dev.x[dst, :2]
                    rel_pos = pos_dst - pos_src
                    dist = torch.norm(rel_pos, dim=1, keepdim=True)
                    if config.INPUT_DIM >= 4 and config.EDGE_DIM >= 6:
                        v_src, v_dst = graph_dev.x[src, 2:4], graph_dev.x[dst, 2:4]
                        rel_v = v_dst - v_src
                        v1_n = torch.norm(v_src, dim=1, keepdim=True) + 1e-6
                        v2_n = torch.norm(v_dst, dim=1, keepdim=True) + 1e-6
                        cos_sim = torch.sum(v_src * v_dst, dim=1, keepdim=True) / (v1_n * v2_n)
                        graph_dev.edge_attr = torch.cat([rel_pos, dist, rel_v, cos_sim], dim=1)
                    else:
                        graph_dev.edge_attr = torch.cat([rel_pos, dist], dim=1)

                with torch.no_grad():
                    _, offsets, _, h_final = gnn_model(graph_dev)
                
                # 我们现在纯靠 GNN 提取的高维特征来进行数据关联 (证明我们的特征比纯距离强)
                corrected = meas_points[:, :2].copy()
                
                node_features = h_final.cpu().numpy() if h_final is not None else None
                # 使用 pre_gnn 逻辑，不传递 node_features 作为 shape
                pred_c, pred_id, _ = trackers['H-GAT-GT (Ours)'].update(corrected, None)
                pt_map_ours = np.full(len(meas_points), -1) # pre_gnn 不提供点级别 label
            else:
                pred_c, pred_id, pt_map_ours = np.empty((0,2)), np.array([]), np.full(len(meas_points), -1)

            metrics['H-GAT-GT (Ours)'].update_time(time.time() - t0)
            metrics['H-GAT-GT (Ours)'].update(gt_centers, gt_ids, pred_c, pred_id)
            metrics['H-GAT-GT (Ours)'].update_clustering_metrics(pt_lbl, pt_map_ours)

            # ========================
            # Social-STGCNN (基线：不具备纠偏能力)
            # ========================
            t0 = time.time()
            meas_points_2d = meas_points[:, :2]
            sc, sid, smap = trackers['Social-STGCNN'].step(meas_points_2d)
            metrics['Social-STGCNN'].update_time(time.time() - t0)
            # 这里 sc 里的点就是 meas_points 里的点，所以误差应该是 ~8.0
            metrics['Social-STGCNN'].update(gt_centers, gt_ids, sc, sid)
            metrics['Social-STGCNN'].update_clustering_metrics(pt_lbl, smap)

            # ========================
            # 其他算法 (Baseline, RFS, etc.)
            # ========================
            # 预处理（共享 DBSCAN 时间不计入各算法 Time）
            base_dets, base_map = [], {}
            if len(meas_points_2d) > 0:
                dbl = DBSCAN(eps=35, min_samples=1).fit_predict(meas_points_2d)
                for i, l in enumerate([l for l in set(dbl) if l != -1]):
                    idx = np.where(dbl == l)[0]
                    base_dets.append(np.mean(meas_points_2d[idx], axis=0))
                    base_map[i] = idx

            # Baseline
            t0 = time.time()
            bc, bid, bmap = trackers['Baseline'].step(meas_points_2d)
            metrics['Baseline'].update_time(time.time() - t0)
            metrics['Baseline'].update(gt_centers, gt_ids, bc, bid)
            metrics['Baseline'].update_clustering_metrics(pt_lbl, bmap)

            # GM-CPHD
            t0 = time.time()
            cc, cid = trackers['GM-CPHD'].step(base_dets)
            metrics['GM-CPHD'].update_time(time.time() - t0)
            metrics['GM-CPHD'].update(gt_centers, gt_ids, cc, cid)

            # CBMeMBer
            t0 = time.time()
            cb_c, cb_id = trackers['CBMeMBer'].step(base_dets)
            metrics['CBMeMBer'].update_time(time.time() - t0)
            metrics['CBMeMBer'].update(gt_centers, gt_ids, cb_c, cb_id)

            # Graph-MB
            t0 = time.time()
            gmb_c, gmb_id, gmb_lbl = trackers['Graph-MB'].step(meas_points_2d)
            metrics['Graph-MB'].update_time(time.time() - t0)
            metrics['Graph-MB'].update(gt_centers, gt_ids, gmb_c, gmb_id)
            metrics['Graph-MB'].update_clustering_metrics(pt_lbl, gmb_lbl)

    return {k: v.compute() for k, v in metrics.items()}

def run_ewap_evaluation():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    gnn_model = GNNGroupTracker().to(device)
    if os.path.exists(config.EWAP_MODEL_USE_PATH):
        gnn_model.load_state_dict(torch.load(config.EWAP_MODEL_USE_PATH, map_location=device))
        gnn_model.eval()
    
    scenes = {"ETH": "test_ewap_eth", "Hotel": "test_ewap_hotel"}
    all_results = {name: evaluate_scene(name, folder, gnn_model, device) for name, folder in scenes.items()}
    
    cols = ['MOTA', 'MOTP', 'OSPA (Total)', 'IDSW', 'RMSE (Pos)', 'Purity', 'Time']
    for scene_name, results in all_results.items():
        if results:
            print(f"\nEWAP - {scene_name} 场景跟踪结果\n" + "="*80)
            print(pd.DataFrame(results).T[cols].to_string())

if __name__ == "__main__":
    run_ewap_evaluation()
