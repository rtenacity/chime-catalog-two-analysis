import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import optuna
import os
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name(0))
class CHIMEFRBDataset(Dataset):
    def __init__(self, hdf5_path, catalog_path, target_length):
        self.hdf5_path = hdf5_path
        self.dt = 0.009830400085775182*1000

        self.target_length = target_length
        cat = pd.read_csv(catalog_path, low_memory=False)
        cat["repeater_name"] = cat["repeater_name"].fillna("").str.strip()
        self.repeater_set = set(
            cat.loc[cat["repeater_name"] != "", "tns_name"].str.strip()
        )
        with h5py.File(hdf5_path, "r") as f:
            self.keys = list(f.keys())

        
    @staticmethod
    def _pad_or_crop(wfall, target_length, center_idx):
        n_freq, n_time = wfall.shape
        half = target_length // 2
        start = center_idx - half
        end = start + target_length
        
        if start >= 0 and end <= n_time:
            return wfall[:, start:end]         
        
        src_start = max(start, 0)
        src_end = min(end, n_time)
        out = np.zeros((n_freq, target_length), dtype=wfall.dtype)
        dst_start = src_start - start
        out[:, dst_start:dst_start + (src_end - src_start)] = wfall[:, src_start:src_end]
        return out
    
    
    def __len__(self):
        return len(self.keys)
    
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        
        with h5py.File(self.hdf5_path, "r") as f:
            wfall = f[key]["wfall_plot"][:]
            extent = np.array(f[key]["extent"])
            
        wfall = wfall.astype(np.float32)
        std = wfall.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        wfall = (wfall - wfall.mean(axis=1, keepdims=True)) / std

        peak = round(-extent[0] / self.dt)
        wfall = self._pad_or_crop(wfall, self.target_length, peak)
        tensor = torch.from_numpy(wfall)
        label = torch.tensor(int(key in self.repeater_set), dtype=torch.long)
        
        return tensor, label

def make_dataloader(
    hdf5_path: str,
    catalog_path: str,
    target_length: int,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    train_frac: float = 0.8,
    seed: int = 42,
):
    dataset = CHIMEFRBDataset(hdf5_path, catalog_path, target_length)
    n_total = len(dataset)

    # Collect labels once up front (needed for stratify)
    labels = [dataset[i][1].item() for i in range(n_total)]

    indices = list(range(n_total))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=1 - train_frac,
        stratify=labels,
        random_state=seed,
    )

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    n_rep = sum(labels)
    n_rep_train = sum(labels[i] for i in train_idx)
    n_rep_val   = sum(labels[i] for i in val_idx)
    print(f"Dataset: {n_total} | repeaters: {n_rep} ({100*n_rep/n_total:.1f}%)")
    print(f"Train: {len(train_idx)} | repeaters: {n_rep_train} ({100*n_rep_train/len(train_idx):.1f}%)")
    print(f"Val:   {len(val_idx)}   | repeaters: {n_rep_val}   ({100*n_rep_val/len(val_idx):.1f}%)")

    return train_loader, val_loader

TARGET_LENGTH = 128

train_loader, val_loader = make_dataloader(
    hdf5_path="/scratch/gpfs/MLISANTI/ra0438/all_bursts.hdf5",
    catalog_path="/home/ra0438/chime-catalog-two-analysis/chimefrbcat2.csv",
    target_length=TARGET_LENGTH,
    batch_size=32,
    num_workers=4,
)


for wfall_batch, label_batch in train_loader:
    print(f"Batch shape : {wfall_batch.shape}")
    print(f"Labels      : {label_batch}")
    break

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] 
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SinusoidalPE(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim)
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe


class FRBMaskedAutoencoder(nn.Module):
    def __init__(self, seq_len, n_freq, embed_dim, contrast_dim=32, mask_ratio=0.25, dropout=0.1, n_heads=2, dim_feedforward=128, n_blocks=2):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = n_freq
        self.embed_dim = embed_dim
        self.contrast_dim = contrast_dim
        self.mask_ratio = mask_ratio

        self.enc_proj = nn.Linear(n_freq, embed_dim)
        self.enc_drop = nn.Dropout(dropout)
        self.enc_pe = SinusoidalPE(seq_len + 1, embed_dim)
        self.enc_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward,
                                       batch_first=True, dropout=dropout) for _ in range(n_blocks)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dec_pe = SinusoidalPE(seq_len + 1, embed_dim)
        self.dec_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, nhead=n_heads, dim_feedforward=dim_feedforward,
                                       batch_first=True, dropout=dropout) for _ in range(n_blocks)
        ])
        self.dec_norm = nn.LayerNorm(embed_dim)
        self.dec_proj = nn.Linear(embed_dim, n_freq)

        self.cls_head = nn.Linear(embed_dim, 1)
        self.cls_drop = nn.Dropout(dropout)
        
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, contrast_dim),
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def mask_input(self, x):
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1,
                                index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def encoder(self, x):
        x = self.enc_drop(self.enc_proj(x))
        x = x + self.enc_pe.pe[1:self.seq_len + 1] 
        x_vis, mask, ids_restore, ids_keep = self.mask_input(x)

        cls = (self.cls_token + self.enc_pe.pe[0]).expand(x_vis.size(0), -1, -1)
        x_vis = torch.cat([cls, x_vis], dim=1)

        for block in self.enc_blocks:
            x_vis = block(x_vis)
        x_vis = self.enc_norm(x_vis)

        return x_vis, mask, ids_restore

    def decoder(self, x_enc, ids_restore):
        B = x_enc.size(0)
        T = ids_restore.size(1)
        n_keep = x_enc.size(1) - 1                      
        mask_tokens = self.mask_token.expand(B, T - n_keep, -1)

        x_no_cls = x_enc[:, 1:, :]
        x_full = torch.cat([x_no_cls, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1,
                              index=ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        x_full = x_full + self.dec_pe.pe[1:T + 1]
        cls = x_enc[:, :1, :] + self.dec_pe.pe[0]
        x_full = torch.cat([cls, x_full], dim=1)

        for block in self.dec_blocks:
            x_full = block(x_full)
        x_full = self.dec_norm(x_full)

        recon = self.dec_proj(x_full[:, 1:, :])
        return recon

    def forward(self, x):
        x_t = x.permute(0, 2, 1)                       
        x_enc, mask, ids_restore = self.encoder(x_t)
        
        cls_token = x_enc[:, 0, :]
        cls_out = self.cls_drop(self.cls_head(cls_token))
        recon = self.decoder(x_enc, ids_restore) 
        
        proj = nn.functional.normalize(self.proj_head(cls_token), dim=1)
        proj = proj.unsqueeze(1)
        return recon, cls_out.squeeze(-1), mask, proj


SupConLoss_fn = SupConLoss(temperature=0.07, contrast_mode='all')

def compute_loss(cls_out, x_recon, mask, wfall, labels, proj,
                 alpha=1.0, beta=1.0, gamma=0.1, pos_weight=None, device='cpu'):
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device) if pos_weight else None
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pw)(cls_out, labels.float())
    target = wfall.permute(0, 2, 1)                   
    diff = (x_recon - target) ** 2
    recon_loss = (diff * mask.unsqueeze(-1)).sum() / (mask.sum() * wfall.size(1))
    con_loss = SupConLoss_fn(proj, labels=labels, mask=None)
    return alpha * cls_loss + beta * recon_loss + gamma * con_loss, cls_loss, recon_loss, con_loss

def train_one_epoch(model, loader, optimiser, device, alpha, beta, gamma, pos_weight_scalar):
    model.train()
    total, correct = 0, 0
    running_loss = running_cls = running_recon = running_con = 0.0

    for wfall, labels in loader:
        wfall, labels = wfall.to(device), labels.to(device)

        x_recon, cls_out, mask, proj = model(wfall)
        loss, cls_loss, recon_loss, con_loss = compute_loss(
            cls_out, x_recon, mask, wfall, labels, proj,
            alpha=alpha, beta=beta, gamma=gamma,
            pos_weight=pos_weight_scalar, device=device
        )

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss  += loss.item()
        running_cls   += cls_loss.item()
        running_recon += recon_loss.item()
        running_con   += con_loss.item()
        
        preds = (torch.sigmoid(cls_out.squeeze(-1)) > 0.5).long()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    print(f"  loss={running_loss/n:.4f}  cls={running_cls/n:.4f}  "
          f"recon={running_recon/n:.4f}  con={running_con/n:.4f}  acc={correct/total:.3f}")


@torch.no_grad()
def evaluate(model, loader, device, alpha, beta, gamma, pos_weight_scalar):
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0
    confusion_matrix_global = np.zeros((2, 2), dtype=int)

    for wfall, labels in loader:
        wfall, labels = wfall.to(device), labels.to(device)
        x_recon, cls_out, mask, proj = model(wfall)
        loss, *_ = compute_loss(
            cls_out, x_recon, mask, wfall, labels, proj,
            alpha=alpha, beta=beta, gamma=gamma,
            pos_weight=pos_weight_scalar, device=device
        )
        running_loss += loss.item()

        preds = (torch.sigmoid(cls_out.squeeze(-1)) > 0.5).long()
        cf_mat = confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])
        confusion_matrix_global += cf_mat
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        

    val_loss = running_loss / len(loader)
    val_acc = correct / total
    return val_loss, val_acc, confusion_matrix_global
N_EPOCHS = 150


CHECKPOINT_DIR = "/scratch/gpfs/MLISANTI/ra0438/cmae_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_overall = {"acc": float('-inf')}

def objective(trial):
    embed_dim       = trial.suggest_categorical("embed_dim",       [32, 64, 128, 256])
    contrast_dim    = trial.suggest_categorical("contrast_dim",    [16, 32, 64])
    mask_ratio      = trial.suggest_float("mask_ratio",            0.1, 0.75)
    dropout         = trial.suggest_float("dropout",               0.0, 0.5)
    n_heads         = trial.suggest_categorical("n_heads",         [1, 2, 4])
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512])
    alpha           = trial.suggest_float("alpha",                 0.1, 5.0)
    beta            = trial.suggest_float("beta",                  0.1, 5.0)
    gamma           = trial.suggest_float("gamma",                 0.01, 1.0, log=True)
    pos_weight      = trial.suggest_float("pos_weight_scalar",     1.0, 5.0)
    lr_patience     = trial.suggest_int("LR_PATIENCE",             3, 15)
    es_patience     = trial.suggest_int("ES_PATIENCE",             5, 25)
    n_blocks        = trial.suggest_categorical("n_blocks",        [2, 4, 6])

    if embed_dim % n_heads != 0:
        raise optuna.exceptions.TrialPruned()

    model = FRBMaskedAutoencoder(
        seq_len=TARGET_LENGTH,
        n_freq=256,
        embed_dim=embed_dim,
        contrast_dim=contrast_dim,
        mask_ratio=mask_ratio,
        dropout=dropout,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        n_blocks=n_blocks
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=lr_patience, min_lr=1e-6
    )

    best_val_acc = float('-inf')
    best_val_loss = float('inf')

    best_state = None
    epochs_no_improve = 0
    
    print(f"Trial {trial.number}: {trial.params}")

    for epoch in range(N_EPOCHS):
        train_one_epoch(model, train_loader, optimiser, device,
                        alpha=alpha, beta=beta, gamma=gamma, pos_weight_scalar=pos_weight)
        val_loss, val_acc, confusion_matrix_global = evaluate(model, val_loader, device,
                            alpha=alpha, beta=beta, gamma=gamma, pos_weight_scalar=pos_weight)
        scheduler.step(val_loss)
        trial.report(val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= es_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
    print(f"Trial {trial.number}: Final val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
    print("  Confusion Matrix:")
    print(confusion_matrix_global)

    trial_path = os.path.join(CHECKPOINT_DIR, f"trial_{trial.number}.pt")
    torch.save({
        "model_state_dict": best_state,
        "params": trial.params,
        "val_loss": best_val_loss,   # was: val_loss
        "val_acc": best_val_acc,
        "trial_number": trial.number,
    }, trial_path)
    print(f"Trial {trial.number} checkpoint saved -> {trial_path}")

    if best_val_acc > best_overall["acc"]:
        best_overall["acc"] = best_val_acc
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save({
            "model_state_dict": best_state,
            "params": trial.params,
            "val_loss": best_val_loss,   # was: val_loss
            "val_acc": best_val_acc,
            "trial_number": trial.number,
        }, best_path)
        print(f"New best model saved -> {best_path}  (val_acc={best_val_acc:.3f}, trial={trial.number})")

    return best_val_acc


study = optuna.create_study(
    direction="maximize",
)
study.optimize(objective, n_trials=50)

print("\nBest trial:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")
    
print(f"\nAll checkpoints saved to: {CHECKPOINT_DIR}")
print(f"Best overall val_acc: {best_overall['acc']:.3f}")