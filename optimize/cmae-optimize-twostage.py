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
#import f1_score
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))


class CHIMEFRBDataset(Dataset):
    def __init__(self, hdf5_path, catalog_path, target_length):
        self.hdf5_path = hdf5_path
        self.dt = 0.009830400085775182*1000
        self._h5 = None

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
    
    def _get_file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    
    def __len__(self):
        return len(self.keys)
    
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        
        f = self._get_file()
        wfall = f[key]["wfall_plot"][:]
        extent = np.array(f[key]["extent"])
            
        wfall = wfall.astype(np.float32)
        std = wfall.std(axis=1, keepdims=True)
        std[std == 0] = 1.0          # avoid divide-by-zero for masked channels
        wfall = (wfall - wfall.mean(axis=1, keepdims=True)) / std
        

        peak = round(-extent[0] / self.dt)
        wfall = self._pad_or_crop(wfall, self.target_length, peak)
        tensor = torch.from_numpy(wfall)
        label = torch.tensor(int(key in self.repeater_set), dtype=torch.long)
        
        return tensor, label


def make_dataloader(hdf5_path: str, catalog_path: str, target_length: int, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, train_frac: float = 0.66, seed: int = 42):
    dataset = CHIMEFRBDataset(hdf5_path, catalog_path, target_length)
    n_total = len(dataset)

    # Collect labels once up front (needed for stratify)
    labels = [dataset[i][1].item() for i in range(n_total)]

    indices = list(range(n_total))
    train_idx, val_idx = train_test_split(indices, test_size=1 - train_frac, stratify=labels, random_state=seed)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    n_rep = sum(labels)
    n_rep_train = sum(labels[i] for i in train_idx)
    n_rep_val = sum(labels[i] for i in val_idx)
    print(f"Dataset: {n_total} | repeaters: {n_rep} ({100*n_rep/n_total:.1f}%)")
    print(f"Train: {len(train_idx)} | repeaters: {n_rep_train} ({100*n_rep_train/len(train_idx):.1f}%)")
    print(f"Val:   {len(val_idx)}   | repeaters: {n_rep_val}   ({100*n_rep_val/len(val_idx):.1f}%)")

    return train_loader, val_loader


TARGET_LENGTH = 128

train_loader, val_loader = make_dataloader(hdf5_path="/scratch/gpfs/MLISANTI/ra0438/all_bursts.hdf5", 
                                           catalog_path="/home/ra0438/chime-catalog-two-analysis/chimefrbcat2.csv", 
                                           train_frac=0.75, target_length=TARGET_LENGTH, batch_size=64, num_workers=5)


for wfall_batch, label_batch in train_loader:
    print(f"Batch shape : {wfall_batch.shape}")
    print(f"Labels      : {label_batch}")
    break


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.eps = 1e-6

    def forward(self, logits, labels):
        p = torch.sigmoid(logits)
        p_t = p * labels + (1 - p) * (1 - labels)
        p_t = p_t.clamp(min=self.eps, max=1 - self.eps)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight, reduction="none"
        )
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class WaterfallAugment(nn.Module):
    def __init__(self, time_mask_frac=0.15, freq_mask_frac=0.15, noise_std=0.05):
        super().__init__()
        self.time_mask_frac = time_mask_frac
        self.freq_mask_frac = freq_mask_frac
        self.noise_std = noise_std

    def forward(self, x):
        x = x.clone()
        B, n_freq, seq_len = x.shape

        # Per-sample time masking
        t_mask = max(1, int(seq_len * self.time_mask_frac))
        for i in range(B):
            t0 = torch.randint(0, seq_len - t_mask, (1,)).item()
            x[i, :, t0 : t0 + t_mask] = 0.0

        # Per-sample frequency masking
        f_mask = max(1, int(n_freq * self.freq_mask_frac))
        for i in range(B):
            f0 = torch.randint(0, n_freq - f_mask, (1,)).item()
            x[i, f0 : f0 + f_mask, :] = 0.0

        # Gaussian noise (already per-sample since randn_like is elementwise)
        x = x + torch.randn_like(x) * self.noise_std

        return x


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
    def __init__(self, seq_len, n_freq, embed_dim, dec_embed_dim, contrast_dim=32, mask_ratio=0.25, dropout=0.1, n_enc_heads=4, n_dec_heads=2, dim_feedforward_enc=128, dim_feedforward_dec=128, n_enc_blocks=2, n_dec_blocks=2):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = n_freq
        self.embed_dim = embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.contrast_dim = contrast_dim
        self.mask_ratio = mask_ratio

        self.enc_proj = nn.Linear(n_freq, embed_dim)
        self.enc_drop = nn.Dropout(dropout)
        self.enc_pe = SinusoidalPE(seq_len + 1, embed_dim)
        self.enc_blocks = nn.ModuleList([nn.TransformerEncoderLayer(embed_dim, 
                                                                    nhead=n_enc_heads, 
                                                                    dim_feedforward=dim_feedforward_enc, 
                                                                    batch_first=True, dropout=dropout) 
                                         for _ in range(n_enc_blocks)])
        self.enc_norm = nn.LayerNorm(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))

        self.enc_to_dec = nn.Linear(embed_dim, dec_embed_dim)
        self.dec_pe = SinusoidalPE(seq_len + 1, dec_embed_dim)
        self.dec_blocks = nn.ModuleList([nn.TransformerEncoderLayer(dec_embed_dim, 
                                                                    nhead=n_dec_heads, 
                                                                    dim_feedforward=dim_feedforward_dec, 
                                                                    batch_first=True, dropout=dropout) 
                                         for _ in range(n_dec_blocks)])
        self.dec_norm = nn.LayerNorm(dec_embed_dim)
        self.dec_proj = nn.Linear(dec_embed_dim, n_freq)

        self.cls_head = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), 
                                      nn.LayerNorm(embed_dim), 
                                      nn.ReLU(),
                                      nn.Dropout(dropout), 
                                      nn.Linear(embed_dim, embed_dim // 2), 
                                      nn.ReLU(), nn.Dropout(dropout), 
                                      nn.Linear(embed_dim // 2, 1))

        self.proj_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), 
                                       nn.LayerNorm(embed_dim), 
                                       nn.ReLU(), 
                                       nn.Dropout(dropout), 
                                       nn.Linear(embed_dim, contrast_dim))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        self.augment_fn = WaterfallAugment(time_mask_frac=0.15, freq_mask_frac=0.15, noise_std=0.05)

    def mask_input(self, x, shared_noise=None):
        B, T, F = x.shape
        len_keep = int(T * (1 - self.mask_ratio))

        if shared_noise is not None:
            noise = shared_noise
        else:
            noise = torch.rand(B, T, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, F))

        mask = torch.ones(B, T, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def encoder(self, x, shared_noise=None, mask_input=True):
        x = self.enc_drop(self.enc_proj(x))
        x = x + self.enc_pe.pe[1 : self.seq_len + 1]

        if mask_input:
            x_vis, mask, ids_restore, ids_keep = self.mask_input(x, shared_noise)
        else:
            x_vis = x
            mask = None
            ids_restore = None
            ids_keep = None

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

        x_enc = self.enc_to_dec(x_enc)

        x_no_cls = x_enc[:, 1:, :]
        x_full = torch.cat([x_no_cls, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.dec_embed_dim))

        x_full = x_full + self.dec_pe.pe[1 : T + 1]
        cls = x_enc[:, :1, :] + self.dec_pe.pe[0]
        x_full = torch.cat([cls, x_full], dim=1)

        for block in self.dec_blocks:
            x_full = block(x_full)
        x_full = self.dec_norm(x_full)

        recon = self.dec_proj(x_full[:, 1:, :])
        return recon

    def forward_pretrain(self, x, augment=True):
        x_t = x.permute(0, 2, 1)
        shared_noise = torch.rand(x_t.size(0), self.seq_len, device=x_t.device)

        x_enc, mask, ids_restore = self.encoder(x_t, shared_noise)
        cls_token = x_enc[:, 0, :]

        recon = self.decoder(x_enc, ids_restore)

        proj1 = nn.functional.normalize(self.proj_head(cls_token), dim=1)

        if augment:
            x_aug = self.augment_fn(x)
            x_aug_t = x_aug.permute(0, 2, 1)
            x_enc2, _, _ = self.encoder(x_aug_t, shared_noise)
            cls_token2 = x_enc2[:, 0, :]
            proj2 = nn.functional.normalize(self.proj_head(cls_token2), dim=1)
            proj = torch.stack([proj1, proj2], dim=1)  # [B, 2, contrast_dim]
        else:
            proj = proj1.unsqueeze(1)  # fallback: [B, 1, contrast_dim]

        return recon, mask, proj

    def forward_finetune(self, x):

        x_t = x.permute(0, 2, 1)
        shared_noise = torch.rand(x_t.size(0), self.seq_len, device=x_t.device)

        x_enc, _, _ = self.encoder(x_t, shared_noise, mask_input=False)

        cls_token = x_enc[:, 0, :]
        seq_tokens = x_enc[:, 1:, :]
        mean_pooled = seq_tokens.mean(dim=1)

        rich_embed = torch.cat([cls_token, mean_pooled], dim=1)

        cls_out = self.cls_head(rich_embed)

        return cls_out.squeeze(-1)


SupConLoss_fn = SupConLoss(temperature=0.07, contrast_mode="all")


def compute_loss(cls_out, x_recon, mask, wfall, labels, proj, alpha=1.0, beta=1.0, gamma=0.1, pos_weight=None, device="cpu"):
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device) if pos_weight else None
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pw)(cls_out, labels.float())
    target = wfall.permute(0, 2, 1)
    diff = (x_recon - target) ** 2
    recon_loss = (diff * mask.unsqueeze(-1)).sum() / (mask.sum() * wfall.size(1))
    con_loss = SupConLoss_fn(proj, labels=labels, mask=None)
    return (alpha * cls_loss + beta * recon_loss + gamma * con_loss, cls_loss, recon_loss, con_loss)


def compute_pretrain_loss(x_recon, mask, wfall, proj, labels, beta=1.0, gamma=0.1):
    target = wfall.permute(0, 2, 1)
    diff = (x_recon - target) ** 2
    recon_loss = (diff * mask.unsqueeze(-1)).sum() / (mask.sum() * wfall.size(1))
    con_loss = SupConLoss_fn(proj, labels=labels, mask=None)
    return beta * recon_loss + gamma * con_loss, recon_loss, con_loss


def compute_finetune_loss(cls_out, labels, pos_weight=None, focal_param=0.0, device="cpu"):
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device) if pos_weight else None
    cls_loss = FocalLoss(gamma=focal_param, pos_weight=pw)(cls_out, labels.float())
    return cls_loss


def pretrain_one_epoch(model, loader, optimizer, device, beta, gamma, pos_weight_scalar):
    model.train()
    running_loss = running_recon = running_con = 0.0

    for wfall, labels in loader:
        wfall, labels = wfall.to(device), labels.to(device)

        recon, mask, proj = model.forward_pretrain(wfall, augment=True)
        loss, recon_loss, con_loss = compute_pretrain_loss(recon, mask, wfall, proj, labels, beta=beta, gamma=gamma)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_recon += recon_loss.item()
        running_con += con_loss.item()

    n = len(loader)
    print(f"  loss={running_loss/n:.4f}  recon={running_recon/n:.4f}  con={running_con/n:.4f}")


def finetune_one_epoch(model, loader, optimizer, device, alpha, pos_weight_scalar, focal_param):
    model.train()
    total, correct = 0, 0
    running_loss = 0.0

    for wfall, labels in loader:
        wfall, labels = wfall.to(device), labels.to(device)

        cls_out = model.forward_finetune(wfall)
        loss = compute_finetune_loss(cls_out, labels, pos_weight=pos_weight_scalar, focal_param=focal_param, device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        preds = (torch.sigmoid(cls_out) > 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    n = len(loader)
    print(f"  loss={running_loss/n:.4f}  acc={correct/total:.3f}")


@torch.no_grad()
def evaluate_pretrain(model, loader, device, beta, gamma, pos_weight_scalar):
    model.eval()
    running_loss = running_recon = running_con = 0.0
    for wfall, labels in loader:
        wfall, labels = wfall.to(device), labels.to(device)
        x_recon, mask, proj = model.forward_pretrain(wfall, augment=True)
        loss, recon_loss, con_loss = compute_pretrain_loss(x_recon, mask, wfall, proj, labels, beta=beta, gamma=gamma)
        running_loss += loss.item()
        running_recon += recon_loss.item()
        running_con += con_loss.item()
    n = len(loader)
    print(f"  val_loss={running_loss/n:.4f}  val_recon={running_recon/n:.4f}  val_con={running_con/n:.4f}")
    return running_loss / n, running_recon / n, running_con / n


@torch.no_grad()
def sweep_threshold(model, loader, device, alpha, beta, gamma, pos_weight_scalar, focal_param):
    model.eval()
    all_probs, all_labels = [], []
    running_loss = 0.0

    for wfall, labels in loader:
        wfall, labels = wfall.to(device), labels.to(device)
        cls_out = model.forward_finetune(wfall)

        loss = compute_finetune_loss(cls_out, labels, pos_weight=pos_weight_scalar, focal_param=focal_param, device=device)
        running_loss += loss.item()

        probs = torch.sigmoid(cls_out).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_thresh_acc, best_acc = 0.5, 0.0
    for thresh in np.linspace(0.05, 0.95, 91):
        preds = (all_probs > thresh).astype(int)
        acc = (preds == all_labels).mean()
        if acc > best_acc:
            best_acc, best_thresh_acc = acc, thresh
            
            
    best_thresh_f1, best_f1 = 0.5, 0.0
    for thresh in np.linspace(0.05, 0.95, 91):
        preds = (all_probs > thresh).astype(int)
        # get f1 score on repeater class (label 1)
        f1 = f1_score(all_labels, preds, pos_label=1)
        if f1 > best_f1:
            best_f1, best_thresh_f1 = f1, thresh

    best_preds = (all_probs > best_thresh_f1).astype(int)
    confusion_matrix_global = np.zeros((2, 2), dtype=int)
    confusion_matrix_global = confusion_matrix(all_labels, best_preds, labels=[0, 1])

    val_loss = running_loss / len(loader)

    print(f"Best val acc   : {best_acc:.4f}")
    print(f"Best threshold : {best_thresh_acc:.2f}")

    print(f"Best f1        : {best_f1:.4f}")
    print(f"Best threshold f1: {best_thresh_f1:.2f}")
    print(f"Confusion matrix (optimized for f1):\n{confusion_matrix_global}")

    return val_loss, best_thresh_acc, best_acc, best_thresh_f1, best_f1, confusion_matrix_global


N_EPOCHS = 250


CHECKPOINT_DIR = "/scratch/gpfs/MLISANTI/ra0438/cmae_checkpoints_f1_v3"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_overall = {"f1": float("-inf")}


def objective(trial):
    lr_patience = 15
    es_patience = 20
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128, 256, 512])
    dec_emb_frac = trial.suggest_categorical("dec_emb_frac", [0.25, 0.5, 1.0])
    dec_emb_dim = int(embed_dim * dec_emb_frac)

    contrast_dim = trial.suggest_categorical("contrast_dim", [16, 32, 64])
    mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.75)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    n_enc_heads = trial.suggest_categorical("n_enc_heads", [1, 2, 4])
    n_dec_head_frac = trial.suggest_categorical("n_dec_frac", [0.25, 0.5, 1.0])
    n_dec_heads = max(1, int(n_enc_heads * n_dec_head_frac))

    dim_feedforward_enc = trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512])
    dim_feedforward_dec_frac = trial.suggest_categorical("dim_feedforward_dec_frac", [0.25, 0.5, 1.0])
    dim_feedforward_dec = int(dim_feedforward_enc * dim_feedforward_dec_frac)

    beta = trial.suggest_float("beta", 0.1, 10.0)
    gamma = trial.suggest_float("gamma", 0.01, 1.0, log=True)
    pos_weight = trial.suggest_float("pos_weight_scalar", 1.0, 5.0)
    focal_param = trial.suggest_float("focal_gamma", -1.0, 3.0)
    n_enc_blocks = trial.suggest_categorical("n_enc_blocks", [2, 4, 6, 8])
    n_dec_block_frac = trial.suggest_categorical("n_dec_block_frac", [0.25, 0.5, 1.0])
    pretrain_frac = trial.suggest_float("pretrain_frac", 0.5, 0.9)

    n_dec_blocks = max(1, int(n_enc_blocks * n_dec_block_frac))

    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    if embed_dim % n_enc_heads != 0 or dec_emb_dim % n_dec_heads != 0:
        raise optuna.exceptions.TrialPruned()

    model = FRBMaskedAutoencoder(seq_len=TARGET_LENGTH, 
                                 n_freq=256, 
                                 embed_dim=embed_dim, 
                                 dec_embed_dim=dec_emb_dim, 
                                 contrast_dim=contrast_dim, 
                                 mask_ratio=mask_ratio, 
                                 dropout=dropout, 
                                 n_enc_heads=n_enc_heads, 
                                 n_dec_heads=n_dec_heads, 
                                 dim_feedforward_enc=dim_feedforward_enc, 
                                 dim_feedforward_dec=dim_feedforward_dec, 
                                 n_enc_blocks=n_enc_blocks, 
                                 n_dec_blocks=n_dec_blocks).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-6)

    best_val_f1 = float("-inf")
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_confusion_matrix = None

    best_state = None
    epochs_no_improve = 0
    best_thresh = 0.5

    print(f"Trial {trial.number}: {trial.params}")

    for epoch in range(int(N_EPOCHS * pretrain_frac)):
        pretrain_one_epoch(model, train_loader, optimizer, device, beta=beta, gamma=gamma, pos_weight_scalar=pos_weight)

        loss, recon_loss, con_loss = evaluate_pretrain(model, val_loader, device, beta=beta, gamma=gamma, pos_weight_scalar=pos_weight)
        scheduler.step(loss)
        print(f"Pretrain Epoch {epoch+1}/{int(N_EPOCHS * pretrain_frac)}: val_loss={loss:.4f}  val_recon={recon_loss:.4f}  val_con={con_loss:.4f}")

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ("enc_proj", "enc_drop", "enc_pe", "enc_blocks", "enc_norm")):
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-6)

    for epoch in range(int(N_EPOCHS * (1 - pretrain_frac))):
        finetune_one_epoch(model, train_loader, optimizer, device, alpha=1.0, pos_weight_scalar=pos_weight, focal_param=focal_param)

        val_loss, opt_thresh_acc, val_acc, opt_thresh, val_f1, confusion_matrix_global = sweep_threshold(model, val_loader, device, alpha=1.0, beta=beta, gamma=gamma, pos_weight_scalar=pos_weight, focal_param=focal_param)

        scheduler.step(val_loss)
        trial.report(val_f1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_f1 > best_val_f1:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_confusion_matrix = confusion_matrix_global
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            best_thresh = opt_thresh
            best_thresh_acc = opt_thresh_acc
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= es_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Trial {trial.number}: Final val_loss={best_val_loss:.4f}, best_thresh (for f1) = {best_thresh:.3f}, val_f1={best_val_f1:.3f}, val_acc={best_val_acc:.3f}, best_thresh_acc={best_thresh_acc:.3f}")
    print("  Confusion Matrix:")
    print(best_confusion_matrix)

    trial_path = os.path.join(CHECKPOINT_DIR, f"trial_{trial.number}.pt")
    torch.save({"model_state_dict": best_state, "params": trial.params, "val_loss": best_val_loss, "val_f1": best_val_f1, "best_thresh": best_thresh,  "val_acc": best_val_acc, "best_thresh_acc": best_thresh_acc, "trial_number": trial.number, "confusion_matrix": best_confusion_matrix}, trial_path)  # was: val_loss
    print(f"Trial {trial.number} checkpoint saved -> {trial_path}")

    if best_val_f1 > best_overall["f1"]:
        best_overall["f1"] = best_val_f1
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save({"model_state_dict": best_state, "params": trial.params, "val_loss": best_val_loss, "val_f1": best_val_f1, "best_thresh": best_thresh, "trial_number": trial.number}, best_path)
        print(f"New best model saved -> {best_path}  (val_f1={best_val_f1:.3f}, trial={trial.number})")

    return best_val_f1


study = optuna.create_study(
    study_name="cmae_optimize",
    storage="sqlite:////scratch/gpfs/MLISANTI/ra0438/cmae_study_f1_v3.db",
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15),
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)

print("\nBest trial:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")

print(f"\nAll checkpoints saved to: {CHECKPOINT_DIR}")
print(f"Best overall val_f1: {best_overall['f1']:.3f}")