import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv, TransformerConv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_ENV_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SIM_ENV_DIR)
for path in (PROJECT_ROOT, SIM_ENV_DIR):
    if path not in sys.path:
        sys.path.append(path)

import config
from sim_env.model import TrackerOutputs


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
        alpha = self.softmax(torch.mean(self.attn_vector, dim=1))
        out = 0
        for i, layer in enumerate(layers_list):
            out = out + layer * alpha[i]
        return out


class AblationGNNTracker(nn.Module):
    def __init__(
        self,
        use_fourier=True,
        fusion_mode='adaptive',
        use_transformer=True,
        hidden_dim=None,
        input_node_dim=None,
        input_edge_dim=None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = config.HIDDEN_DIM
        if input_node_dim is None:
            input_node_dim = config.INPUT_DIM
        if input_edge_dim is None:
            input_edge_dim = config.EDGE_DIM
        if fusion_mode not in {'adaptive', 'last'}:
            raise ValueError(f'Unsupported fusion_mode: {fusion_mode}')

        self.use_fourier = use_fourier
        self.use_transformer = use_transformer
        self.fusion_mode = fusion_mode
        self.hidden_dim = hidden_dim

        self.fourier_dim = 64 if use_fourier else 0
        self.pos_encoder = None
        if use_fourier:
            self.pos_encoder = FourierFeatureEncoder(input_node_dim, self.fourier_dim // 2, scale=2.0)

        self.node_mlp = nn.Sequential(
            nn.Linear(input_node_dim + self.fourier_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(4):
            if use_transformer:
                conv = TransformerConv(
                    hidden_dim,
                    hidden_dim // 4,
                    heads=4,
                    edge_dim=hidden_dim,
                    dropout=0.1,
                )
            else:
                conv = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
            self.norms.append(BatchNorm(hidden_dim))

        self.fusion_layer = AdaptiveLayerFusion(hidden_dim, num_layers=4) if fusion_mode == 'adaptive' else None

        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.group_offset_regressor = self._build_offset_head(hidden_dim)
        self.point_offset_regressor = self._build_offset_head(hidden_dim)

    def _build_offset_head(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        expected_edge_dim = self.edge_encoder[0].in_features
        if edge_attr.shape[0] == 0 and edge_attr.shape[1] != expected_edge_dim:
            edge_attr = torch.empty((0, expected_edge_dim), dtype=x.dtype, device=x.device)

        if self.use_fourier:
            x_in = torch.cat([x, self.pos_encoder(x)], dim=1)
        else:
            x_in = x
        h_in = self.node_mlp(x_in)
        e = self.edge_encoder(edge_attr)

        layers = []
        for conv, norm in zip(self.convs, self.norms):
            if self.use_transformer:
                h_out = conv(h_in, edge_index, edge_attr=e)
            else:
                h_out = conv(h_in, edge_index)
            h_out = norm(h_out)
            h_out = F.gelu(h_out)
            h_out = h_out + h_in
            layers.append(h_out)
            h_in = h_out

        h_final = self.fusion_layer(layers) if self.fusion_layer is not None else layers[-1]

        row, col = edge_index
        edge_feat = torch.cat([h_final[row], h_final[col], e], dim=1)
        edge_scores = self.edge_classifier(edge_feat).squeeze(-1)

        group_out = self.group_offset_regressor(h_final)
        group_offsets = group_out[:, :2]
        group_uncertainty = F.softplus(group_out[:, 2:]) + 1e-6

        point_out = self.point_offset_regressor(h_final)
        point_offsets = point_out[:, :2]
        point_uncertainty = F.softplus(point_out[:, 2:]) + 1e-6

        return TrackerOutputs(
            edge_scores=edge_scores,
            group_offsets=group_offsets,
            group_uncertainty=group_uncertainty,
            point_offsets=point_offsets,
            point_uncertainty=point_uncertainty,
        )
