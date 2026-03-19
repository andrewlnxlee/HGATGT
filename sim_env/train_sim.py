# train.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import config
from dataset import RadarFileDataset
from model import GNNGroupTracker


def build_lookup(table):
    lookup = {}
    if table.numel() == 0:
        return lookup
    if table.dim() == 1:
        table = table.unsqueeze(0)
    for row in table:
        key = int(row[0].item())
        lookup[key] = row[1:3]
    return lookup


def extract_targets(node_ids, gt_table, meas_points):
    gt_lookup = build_lookup(gt_table)
    if len(gt_lookup) == 0 or meas_points.shape[0] == 0:
        return None

    valid_indices = []
    valid_ids = []
    target_positions = []
    for idx, raw_id in enumerate(node_ids):
        target_id = int(raw_id.item())
        if target_id <= 0 or target_id not in gt_lookup:
            continue
        valid_indices.append(idx)
        valid_ids.append(target_id)
        target_positions.append(gt_lookup[target_id])

    if not valid_indices:
        return None

    device = meas_points.device
    indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
    ids = torch.tensor(valid_ids, dtype=torch.long, device=device)
    positions = torch.stack(target_positions).to(device)
    offsets = positions - meas_points[indices]
    return {
        'indices': indices,
        'ids': ids,
        'positions': positions,
        'offsets': offsets,
    }


def heteroscedastic_nll(pred_mu, pred_sigma, target):
    variance = pred_sigma.pow(2).clamp_min(1e-6)
    mse = (pred_mu - target).pow(2)
    return 0.5 * (mse / variance + torch.log(variance)).mean()


def compute_edge_loss(pred_scores, data):
    return F.binary_cross_entropy(pred_scores, data.edge_label)


def compute_regression_loss(pred_offsets, pred_uncertainty, targets):
    if targets is None:
        return pred_offsets.new_tensor(0.0)
    pred_mu = pred_offsets[targets['indices']]
    pred_sigma = pred_uncertainty[targets['indices']]
    return heteroscedastic_nll(pred_mu, pred_sigma, targets['offsets'])


def compute_point_temporal_loss(point_corrected, point_targets, prev_point_state):
    if point_targets is None or not prev_point_state:
        return point_corrected.new_tensor(0.0)

    corrected_valid = point_corrected[point_targets['indices']]
    pred_motion = []
    gt_motion = []

    for idx, point_id in enumerate(point_targets['ids'].detach().cpu().tolist()):
        if point_id not in prev_point_state:
            continue
        prev_state = prev_point_state[point_id]
        pred_motion.append(corrected_valid[idx] - prev_state['corrected_pos'])
        gt_motion.append(point_targets['positions'][idx] - prev_state['gt_position'])

    if not pred_motion:
        return point_corrected.new_tensor(0.0)

    pred_motion = torch.stack(pred_motion)
    gt_motion = torch.stack(gt_motion)
    return F.smooth_l1_loss(pred_motion, gt_motion)


def build_prev_point_state(point_corrected, point_targets):
    if point_targets is None:
        return {}

    corrected_valid = point_corrected[point_targets['indices']].detach()
    gt_positions = point_targets['positions'].detach()
    point_ids = point_targets['ids'].detach().cpu().tolist()

    return {
        point_id: {
            'corrected_pos': corrected_valid[idx],
            'gt_position': gt_positions[idx],
        }
        for idx, point_id in enumerate(point_ids)
    }


def compute_frame_loss(outputs, data, prev_point_state=None):
    if prev_point_state is None:
        prev_point_state = {}

    if (config.LAMBDA_POINT > 0 or config.LAMBDA_TEMP > 0) and (
        not getattr(data, 'has_gt_points', False) or not getattr(data, 'has_point_ids', False)
    ):
        raise ValueError('训练样本缺少 gt_points/point_ids，无法进行点级双分支训练；请先重生成 sim_env 数据。')

    loss_edge = compute_edge_loss(outputs.edge_scores, data)

    group_targets = extract_targets(data.point_labels, data.gt_centers, data.x)
    point_targets = extract_targets(data.point_ids, data.gt_points, data.x)

    loss_group = compute_regression_loss(outputs.group_offsets, outputs.group_uncertainty, group_targets)
    loss_point = compute_regression_loss(outputs.point_offsets, outputs.point_uncertainty, point_targets)

    point_corrected = data.x + outputs.point_offsets
    loss_temp = compute_point_temporal_loss(point_corrected, point_targets, prev_point_state)

    loss = (
        loss_edge
        + config.LAMBDA_GROUP * loss_group
        + config.LAMBDA_POINT * loss_point
        + config.LAMBDA_TEMP * loss_temp
    )

    next_prev_point_state = build_prev_point_state(point_corrected, point_targets)
    stats = {
        'edge': float(loss_edge.detach().item()),
        'group': float(loss_group.detach().item()),
        'point': float(loss_point.detach().item()),
        'temp': float(loss_temp.detach().item()),
        'total': float(loss.detach().item()),
    }
    return loss, stats, next_prev_point_state


def train():
    device = torch.device(config.DEVICE)
    print('Loading datasets...')
    train_set = RadarFileDataset('train')
    val_set = RadarFileDataset('val')

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=0)

    model = GNNGroupTracker(
        input_node_dim=config.INPUT_DIM,
        input_edge_dim=config.EDGE_DIM,
        hidden_dim=config.HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    accumulation_steps = 8

    best_val_loss = float('inf')
    print(f'Start Training Multi-Scale Graph Transformer on {device}...')

    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        total_components = {'edge': 0.0, 'group': 0.0, 'point': 0.0, 'temp': 0.0}
        frame_count = 0
        current_batch_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.EPOCHS} [Train]')
        for episode_graphs in pbar:
            prev_point_state = {}
            for graph in episode_graphs:
                graph = graph.to(device)
                if graph.edge_index.shape[1] == 0:
                    prev_point_state = {}
                    continue

                outputs = model(graph)
                loss, stats, prev_point_state = compute_frame_loss(outputs, graph, prev_point_state)
                (loss / accumulation_steps).backward()

                current_batch_loss += stats['total']
                total_loss += stats['total']
                for key in total_components:
                    total_components[key] += stats[key]
                frame_count += 1

                if frame_count % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    current_batch_loss = 0.0

            if frame_count > 0:
                avg_total = total_loss / frame_count
                avg_point = total_components['point'] / frame_count
                avg_group = total_components['group'] / frame_count
                avg_temp = total_components['temp'] / frame_count
                pbar.set_postfix({
                    'loss': f'{avg_total:.4f}',
                    'point': f'{avg_point:.4f}',
                    'group': f'{avg_group:.4f}',
                    'temp': f'{avg_temp:.4f}',
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                })

        if frame_count % accumulation_steps != 0 and frame_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_components = {'edge': 0.0, 'group': 0.0, 'point': 0.0, 'temp': 0.0}
        val_frames = 0
        with torch.no_grad():
            for episode_graphs in val_loader:
                prev_point_state = {}
                for graph in episode_graphs:
                    graph = graph.to(device)
                    if graph.edge_index.shape[1] == 0:
                        prev_point_state = {}
                        continue

                    outputs = model(graph)
                    loss, stats, prev_point_state = compute_frame_loss(outputs, graph, prev_point_state)
                    val_loss += stats['total']
                    for key in val_components:
                        val_components[key] += stats[key]
                    val_frames += 1

        avg_val_loss = val_loss / max(1, val_frames)
        avg_val_group = val_components['group'] / max(1, val_frames)
        avg_val_point = val_components['point'] / max(1, val_frames)
        avg_val_temp = val_components['temp'] / max(1, val_frames)
        print(
            f'Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} '
            f'| Group: {avg_val_group:.4f} | Point: {avg_val_point:.4f} | Temp: {avg_val_temp:.4f}'
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print('  >>> Best model saved.')


if __name__ == '__main__':
    if not os.path.exists(config.DATA_ROOT):
        print('Error: Data not found.')
    else:
        train()
