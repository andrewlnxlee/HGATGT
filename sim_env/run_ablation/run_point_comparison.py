import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_ENV_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SIM_ENV_DIR)
for path in (PROJECT_ROOT, SIM_ENV_DIR):
    if path not in sys.path:
        sys.path.append(path)

import config
from run_ablation.ablation_model import AblationGNNTracker
from sim_env.dataset import RadarFileDataset
from sim_env.evaluate_single import (
    GroupConstrainedPointAssociator,
    create_metrics,
    ensure_point_gt,
    extract_detected_point_gt,
    filter_clustered_points,
    infer_corrected_points,
    run_hgat_point_identity_pipeline,
)
from sim_env.model import GNNGroupTracker
from trackers.gnn_processor import GNNPostProcessor


CORE_COLUMNS = ['OSPA (Total)', 'RMSE (Pos)', 'IDSW', 'MOTA', 'MOTP', 'FAR', 'Time']


def infer_input_dims(dataset):
    for episode in dataset:
        for graph in episode:
            if graph.x.shape[0] == 0:
                continue
            node_dim = graph.x.shape[1]
            edge_dim = graph.edge_attr.shape[1] if graph.edge_attr.dim() == 2 else config.EDGE_DIM
            return node_dim, edge_dim
    return config.INPUT_DIM, config.EDGE_DIM


def ablation_checkpoint(name):
    return os.path.join(PROJECT_ROOT, 'sim_env', 'run_ablation', 'model', f'model_{name}.pth')


def load_model(model, weight_path, device, label):
    if not os.path.exists(weight_path):
        print(f'Warning: checkpoint not found for {label}: {weight_path}')
        return None

    state_dict = torch.load(weight_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        legacy_checkpoint = any(key.startswith('offset_regressor.') or key.startswith('bns.') for key in state_dict.keys())
        if legacy_checkpoint:
            print(
                f'Warning: {label} checkpoint is a legacy group-level/single-head weight and cannot be used '
                'for the new point-level ablation pipeline. Retrain this variant with '
                'sim_env/run_ablation/train_ablation.py, then rerun this script.'
            )
        else:
            first_line = str(exc).splitlines()[0]
            print(f'Warning: failed to load {label} checkpoint from {weight_path}: {first_line}')
        return None

    model.to(device).eval()
    return model


def evaluate_variant(name, model, test_set, device):
    metrics = create_metrics()

    for episode_idx in tqdm(range(len(test_set)), desc=f'Eval {name}', leave=False):
        episode_graphs = test_set.get(episode_idx)
        group_tracker = GNNPostProcessor()
        point_assoc = GroupConstrainedPointAssociator()
        metrics.reset_sequence()

        for frame_idx, graph in enumerate(episode_graphs):
            ensure_point_gt(graph, episode_idx, frame_idx)
            gt_pos, gt_ids = extract_detected_point_gt(graph, episode_idx, frame_idx)
            meas_points = graph.x.cpu().numpy()

            t0 = time.time()
            corrected_pos = infer_corrected_points(model, graph, meas_points, device, head='point')
            filtered_points, _, _ = filter_clustered_points(corrected_pos)
            pred_pos, pred_ids = run_hgat_point_identity_pipeline(filtered_points, group_tracker, point_assoc)
            metrics.update_time(time.time() - t0)
            metrics.update(gt_pos, gt_ids, pred_pos, pred_ids)

    return metrics.compute()


def build_variants(input_node_dim, input_edge_dim, device):
    variants = []

    full_model_path = os.path.join(PROJECT_ROOT, config.MODEL_USE_PATH)
    full_model = GNNGroupTracker(
        input_node_dim=input_node_dim,
        input_edge_dim=input_edge_dim,
        hidden_dim=config.HIDDEN_DIM,
    )
    full_model = load_model(full_model, full_model_path, device, 'Full_Model')
    if full_model is not None:
        variants.append(('Full_Model', full_model))

    ablation_defs = {
        'No_Fourier': {'use_fourier': False, 'fusion_mode': 'adaptive', 'use_transformer': True},
        'No_Adaptive_Fusion': {'use_fourier': True, 'fusion_mode': 'last', 'use_transformer': True},
        'Plain_GCN': {'use_fourier': True, 'fusion_mode': 'adaptive', 'use_transformer': False},
    }
    for name, kwargs in ablation_defs.items():
        model = AblationGNNTracker(
            **kwargs,
            input_node_dim=input_node_dim,
            input_edge_dim=input_edge_dim,
            hidden_dim=config.HIDDEN_DIM,
        )
        model = load_model(model, ablation_checkpoint(name), device, name)
        if model is not None:
            variants.append((name, model))

    return variants


def main():
    device_name = config.DEVICE if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    test_set = RadarFileDataset('test', include_empty=True)
    if len(test_set) == 0:
        print('No test data found.')
        return

    input_node_dim, input_edge_dim = infer_input_dims(test_set)
    variants = build_variants(input_node_dim, input_edge_dim, device)
    if not variants:
        print('No checkpoints available for point-level ablation comparison.')
        return

    results = {}
    for name, model in variants:
        results[name] = evaluate_variant(name, model, test_set, device)

    df = pd.DataFrame(results).T
    df = df[[col for col in CORE_COLUMNS if col in df.columns]]

    output_dir = os.path.join(PROJECT_ROOT, 'sim_env', 'run_ablation', 'output', 'result')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ablation_point_comparison.csv')
    df.to_csv(output_path)

    print('\n' + '=' * 100)
    print('POINT-LEVEL ABLATION COMPARISON')
    print('=' * 100)
    print(df.to_string())
    print('=' * 100)
    print(f'Saved to: {output_path}')


if __name__ == '__main__':
    main()
