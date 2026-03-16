import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, LayerNorm as GNNLayerNorm
import numpy as np
import config

class FourierFeatureEncoder(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    def forward(self, x):
        x_proj = (2 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class AdaptiveLayerFusion(nn.Module):
    def __init__(self, hidden_dim, num_layers=4):
        super().__init__()
        self.attn_vector = nn.Parameter(torch.randn(num_layers, hidden_dim))
        self.softmax = nn.Softmax(dim=0)
    def forward(self, layers_list):
        stack = torch.stack(layers_list, dim=0)
        alpha = self.softmax(torch.mean(self.attn_vector, dim=1))
        out = 0
        for i in range(len(layers_list)):
            out += layers_list[i] * alpha[i]
        return out

class AblationGNNTracker(nn.Module):
    def __init__(self, use_fourier=True, fusion_mode='adaptive', use_transformer=True, hidden_dim=96, input_node_dim=None, input_edge_dim=None):
        super(AblationGNNTracker, self).__init__()
        self.cfg = {'fourier': use_fourier, 'fusion': fusion_mode, 'transformer': use_transformer}

        if input_node_dim is None: input_node_dim = config.INPUT_DIM
        if input_edge_dim is None: input_edge_dim = config.EDGE_DIM

        self.fourier_dim = 64 if use_fourier else 0
        if use_fourier: self.pos_encoder = FourierFeatureEncoder(2, 32, scale=2.0)
        
        self.node_mlp = nn.Sequential(
            nn.Linear(input_node_dim + self.fourier_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 修复参数传递问题
        self.convs = nn.ModuleList()
        for _ in range(4):
            if use_transformer:
                conv = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim)
            else:
                conv = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
            
        self.bns = nn.ModuleList([GNNLayerNorm(hidden_dim) for _ in range(4)])
        
        if fusion_mode == 'adaptive': self.fusion_layer = AdaptiveLayerFusion(hidden_dim, num_layers=4)
        
        self.edge_classifier = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.offset_regressor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 兼容处理无边情况: 防止 dataset 这里因为 hardcode 返回了不符合本维度的空 tensor (比如0x6而要求0x3)
        expected_edge_dim = self.edge_encoder[0].in_features
        if edge_attr.shape[0] == 0 and edge_attr.shape[1] != expected_edge_dim:
            edge_attr = torch.empty((0, expected_edge_dim), dtype=x.dtype, device=x.device)
            
        if self.cfg['fourier']:
            x_pos = x[:, :2]
            h = self.node_mlp(torch.cat([x, self.pos_encoder(x_pos)], dim=1))
        else:
            h = self.node_mlp(x)
        e = self.edge_encoder(edge_attr)
        
        layers = []
        h_in = h
        for conv, bn in zip(self.convs, self.bns):
            # GCN 也不支持 edge_attr 参数
            if self.cfg['transformer']:
                h_out = conv(h_in, edge_index, edge_attr=e)
            else:
                h_out = conv(h_in, edge_index)
                
            h_out = F.gelu(bn(h_out))
            h_out = h_out + h_in # Residual
            layers.append(h_out)
            h_in = h_out
            
        h_final = self.fusion_layer(layers) if self.cfg['fusion'] == 'adaptive' else layers[-1]
        
        row, col = edge_index
        edge_scores = self.edge_classifier(torch.cat([h_final[row], h_final[col], e], dim=1)).squeeze(-1)
        
        reg_out = self.offset_regressor(h_final)
        pred_offsets = torch.clamp(reg_out[:, :2], -100, 100)
        pred_uncertainty = torch.clamp(F.softplus(reg_out[:, 2:]), 1e-2, 10.0) 
            
        return edge_scores, pred_offsets, pred_uncertainty, h_final
