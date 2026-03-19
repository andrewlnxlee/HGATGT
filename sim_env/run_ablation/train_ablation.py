import os
import sys

import torch
from torch_geometric.loader import DataLoader
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
from sim_env.train_sim import compute_frame_loss


ABLATION_EPOCHS = 10
TRAIN_SUBSET_SIZE = 500


def infer_input_dims(dataset):
    for episode in dataset:
        for graph in episode:
            if graph.x.shape[0] == 0:
                continue
            node_dim = graph.x.shape[1]
            edge_dim = graph.edge_attr.shape[1] if graph.edge_attr.dim() == 2 else config.EDGE_DIM
            return node_dim, edge_dim
    return config.INPUT_DIM, config.EDGE_DIM


def build_loader(dataset, shuffle=False, subset_size=None):
    if subset_size is not None:
        indices = list(range(min(len(dataset), subset_size)))
        dataset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=lambda x: x[0], num_workers=0)


def checkpoint_path(name):
    return os.path.join(PROJECT_ROOT, 'sim_env', 'run_ablation', 'model', f'model_{name}.pth')


def run_epoch(model, loader, device, optimizer=None, desc='Train'):
    is_train = optimizer is not None
    if is_train:
        model.train()
        grad_context = torch.enable_grad()
    else:
        model.eval()
        grad_context = torch.no_grad()

    total_loss = 0.0
    frame_count = 0
    components = {'edge': 0.0, 'group': 0.0, 'point': 0.0, 'temp': 0.0}

    with grad_context:
        pbar = tqdm(loader, desc=desc, leave=False)
        for episode_graphs in pbar:
            prev_point_state = {}
            for graph in episode_graphs:
                graph = graph.to(device)
                if graph.x.shape[0] == 0 or graph.edge_index.shape[1] == 0:
                    prev_point_state = {}
                    continue

                outputs = model(graph)
                loss, stats, prev_point_state = compute_frame_loss(outputs, graph, prev_point_state)
                if torch.isnan(loss) or torch.isinf(loss):
                    prev_point_state = {}
                    continue

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += stats['total']
                for key in components:
                    components[key] += stats[key]
                frame_count += 1

            if frame_count > 0:
                pbar.set_postfix({
                    'loss': f'{total_loss / frame_count:.4f}',
                    'group': f'{components["group"] / frame_count:.4f}',
                    'point': f'{components["point"] / frame_count:.4f}',
                    'temp': f'{components["temp"] / frame_count:.4f}',
                })

    avg_loss = total_loss / max(1, frame_count)
    avg_components = {key: value / max(1, frame_count) for key, value in components.items()}
    return avg_loss, avg_components


def run_experiment(name, model_config, skip_train=False):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'\n>>> Running {name} on {device}')

    train_set = RadarFileDataset('train')
    val_set = RadarFileDataset('val')
    input_node_dim, input_edge_dim = infer_input_dims(val_set if len(val_set) > 0 else train_set)

    model = AblationGNNTracker(
        **model_config,
        input_node_dim=input_node_dim,
        input_edge_dim=input_edge_dim,
        hidden_dim=config.HIDDEN_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    ckpt_path = checkpoint_path(name)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    if skip_train:
        print(f'Skip training for {name}; expected checkpoint: {ckpt_path}')
        return None

    train_loader = build_loader(train_set, shuffle=True, subset_size=TRAIN_SUBSET_SIZE)
    val_loader = build_loader(val_set, shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(ABLATION_EPOCHS):
        train_loss, train_components = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            desc=f'{name} Epoch {epoch + 1}/{ABLATION_EPOCHS} [Train]',
        )
        val_loss, val_components = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            desc=f'{name} Epoch {epoch + 1}/{ABLATION_EPOCHS} [Val]',
        )

        print(
            f'Epoch {epoch + 1}: '
            f'Train={train_loss:.4f} '
            f'(group={train_components["group"]:.4f}, point={train_components["point"]:.4f}, temp={train_components["temp"]:.4f}) | '
            f'Val={val_loss:.4f} '
            f'(group={val_components["group"]:.4f}, point={val_components["point"]:.4f}, temp={val_components["temp"]:.4f})'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f'  >>> Best model saved to {ckpt_path}')

    return best_val_loss


if __name__ == '__main__':
    experiments = {
        'No_Fourier': {'use_fourier': False, 'fusion_mode': 'adaptive', 'use_transformer': True},
        'No_Adaptive_Fusion': {'use_fourier': True, 'fusion_mode': 'last', 'use_transformer': True},
        'Plain_GCN': {'use_fourier': True, 'fusion_mode': 'adaptive', 'use_transformer': False},
    }

    summary = {}
    for name, cfg in experiments.items():
        summary[name] = run_experiment(name, cfg)

    print('\n' + '=' * 48)
    print('Ablation Summary (Best Val Loss)')
    print('=' * 48)
    for name, value in summary.items():
        if value is None:
            print(f'{name:<24}: skipped')
        else:
            print(f'{name:<24}: {value:.4f}')
    print('=' * 48)
