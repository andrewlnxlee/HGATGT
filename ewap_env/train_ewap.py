# train_eth_scratch.py
import os
# 添加根目录路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset as PyGDataset
from tqdm import tqdm

class ETHScratchDataset(RadarFileDataset):
    def __init__(self, split='train'):
        # 覆写路径，使用 finetune 的数据目录 (虽然名字叫finetune，但里面存的是ETH伪标签数据)
        root_dir = os.path.join(config.DATA_ROOT, f"finetune_{split}")
        PyGDataset.__init__(self, root_dir)
        self.root_dir = root_dir
        self.file_list = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.npy')])
        self.conn_radius = 30.0

def compute_loss(pred_scores, pred_offsets, pred_uncertainty, data):
    num_pos = data.edge_label.sum().item()
    num_neg = data.edge_label.numel() - num_pos
    if num_pos > 0:
        weight_factor = min(num_neg / num_pos, 10.0) 
    else:
        weight_factor = 1.0
        
    loss_edge = F.binary_cross_entropy(pred_scores, data.edge_label, weight=None)
    
    id_map = {}
    if data.gt_centers.dim() > 1:
        for row in data.gt_centers:
            gid = int(row[0].item())
            id_map[gid] = row[1:3]
            
    target_offsets = []
    valid_mask = []
    
    for i, uid in enumerate(data.point_labels):
        uid = int(uid.item())
        if uid != 0 and uid in id_map:
            # 仅使用坐标前2维进行偏移计算
            target = id_map[uid] - data.x[i, :2]
            target_offsets.append(target)
            valid_mask.append(i)
            
    if len(valid_mask) > 0:
        target = torch.stack(target_offsets).to(pred_offsets.device)
        pred_mu = pred_offsets[valid_mask]
        pred_sigma = pred_uncertainty[valid_mask] 
        variance = pred_sigma.pow(2)
        mse = (pred_mu - target).pow(2)
        loss_nll = 0.5 * (mse / (variance + 1e-6) + torch.log(variance + 1e-6)).mean()
        loss_reg = loss_nll
    else:
        loss_reg = torch.tensor(0.0).to(pred_scores.device)
        
    return loss_edge + 1.0 * loss_reg, loss_edge.item(), loss_reg.item()

def train_scratch():
    device = torch.device(config.DEVICE)
    print("Loading ETH-only Pseudo Dataset for SCRATCH training...")
    train_set = ETHScratchDataset('train')
    val_set = ETHScratchDataset('val')
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=0)
    
    # 重新初始化模型，不加载任何预训练权重
    model = GNNGroupTracker().to(device)
    print("Model initialized from scratch (no pre-trained weights).")
    
    # 使用标准训练的学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    accumulation_steps = 8  
    best_val_loss = float('inf')
    epochs = 100 # 从头训练需要更多的 Epoch
    
    model_path = "best_model_eth_scratch.pth"
    
    print(f"Start Training Multi-Scale Graph Transformer on ETH Dataset from Scratch ({device})...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad() 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Scratch]")
        
        step_count = 0
        current_batch_loss = 0
        
        for i, episode_graphs in enumerate(pbar):
            for graph in episode_graphs:
                graph = graph.to(device)
                if graph.edge_index.shape[1] == 0: continue 
                
                scores, offsets, uncertainty, _ = model(graph)
                loss, l_edge, l_reg = compute_loss(scores, offsets, uncertainty, graph)
                
                loss = loss / accumulation_steps 
                loss.backward()
                
                current_batch_loss += loss.item() * accumulation_steps
                step_count += 1
                
                if (step_count + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += current_batch_loss
                    current_batch_loss = 0
            
            if step_count > 0:
                pbar.set_postfix({'avg_loss': f"{total_loss/step_count:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        if step_count % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for episode_graphs in val_loader:
                for graph in episode_graphs:
                    graph = graph.to(device)
                    if graph.edge_index.shape[1] == 0: continue
                    scores, offsets, uncertainty, _ = model(graph)
                    loss, _, _ = compute_loss(scores, offsets, uncertainty, graph)
                    val_loss += loss.item()
                    val_steps += 1
                    
        avg_val_loss = val_loss / max(1, val_steps)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  >>> Best model saved to {model_path}")

if __name__ == "__main__":
    train_scratch()
