import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


class CHIMEFRBDataset(Dataset):
    def __init__(self, hdf5_path, catalog_path, target_length):
        self.hdf5_path = hdf5_path
        self.dt = 0.009830400085775182 * 1000
        self._h5 = None

        self.target_length = target_length
        cat = pd.read_csv(catalog_path, low_memory=False)
        cat["repeater_name"] = cat["repeater_name"].fillna("").str.strip()
        self.repeater_set = set(
            cat.loc[cat["repeater_name"] != "", "tns_name"].str.strip()
        )
        with h5py.File(hdf5_path, "r") as f:
            self.keys = list(f.keys())

        self.labels = np.array([int(k in self.repeater_set) for k in self.keys], dtype=np.int64)

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
        std[std == 0] = 1.0  # avoid divide-by-zero for masked channels
        wfall = (wfall - wfall.mean(axis=1, keepdims=True)) / std

        peak = round(-extent[0] / self.dt)
        wfall = self._pad_or_crop(wfall, self.target_length, peak)
        tensor = torch.from_numpy(wfall)
        label = torch.tensor(int(key in self.repeater_set), dtype=torch.long)

        return tensor, label


TARGET_LENGTH = 128 
N_FREQ_CHANNELS = 256
N_SPLITS = 5
BATCH_SIZE = 64
NUM_WORKERS = 5
SEED = 42

dataset = CHIMEFRBDataset(
    hdf5_path="/scratch/gpfs/MLISANTI/ra0438/all_bursts.hdf5",
    catalog_path="/home/ra0438/chime-catalog-two-analysis/chimefrbcat2.csv",
    target_length=TARGET_LENGTH,
)

n_total = len(dataset)
n_rep = int(dataset.labels.sum())
print(f"Dataset: {n_total} | repeaters: {n_rep} ({100 * n_rep / n_total:.1f}%)")


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
        B, F, T = x.shape
        len_keep = int(F * (1 - self.mask_ratio))

        if shared_noise is not None:
            noise = shared_noise
        else:
            noise = torch.rand(B, F, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, T))

        mask = torch.ones(B, F, device=x.device)
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
        x_t = x
        shared_noise = torch.rand(x_t.size(0), self.seq_len, device=x_t.device)

        x_enc, mask, ids_restore = self.encoder(x_t, shared_noise)
        cls_token = x_enc[:, 0, :]

        recon = self.decoder(x_enc, ids_restore)

        proj1 = nn.functional.normalize(self.proj_head(cls_token), dim=1)

        if augment:
            x_aug = self.augment_fn(x)
            x_enc2, _, _ = self.encoder(x_aug, shared_noise)
            cls_token2 = x_enc2[:, 0, :]
            proj2 = nn.functional.normalize(self.proj_head(cls_token2), dim=1)
            proj = torch.stack([proj1, proj2], dim=1)  # [B, 2, contrast_dim]
        else:
            proj = proj1.unsqueeze(1)  # fallback: [B, 1, contrast_dim]

        return recon, mask, proj

    def forward_finetune(self, x):
        x_t = x  
        shared_noise = torch.rand(x_t.size(0), self.seq_len, device=x_t.device)

        x_enc, _, _ = self.encoder(x_t, shared_noise, mask_input=False)

        cls_token = x_enc[:, 0, :]
        seq_tokens = x_enc[:, 1:, :]
        mean_pooled = seq_tokens.mean(dim=1)

        rich_embed = torch.cat([cls_token, mean_pooled], dim=1)

        cls_out = self.cls_head(rich_embed)

        return cls_out.squeeze(-1)


SupConLoss_fn = SupConLoss(temperature=0.07, contrast_mode="all")


def compute_pretrain_loss(x_recon, mask, wfall, proj, labels, beta=1.0, gamma=0.1):
    target = wfall
    diff = (x_recon - target) ** 2
    recon_loss = (diff * mask.unsqueeze(-1)).sum() / (mask.sum() * target.size(-1))
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
        f1 = f1_score(all_labels, preds, pos_label=1)
        if f1 > best_f1:
            best_f1, best_thresh_f1 = f1, thresh

    best_preds = (all_probs > best_thresh_f1).astype(int)
    confusion_matrix_global = confusion_matrix(all_labels, best_preds, labels=[0, 1])

    val_loss = running_loss / len(loader)

    print(f"Best val acc   : {best_acc:.4f}")
    print(f"Best threshold : {best_thresh_acc:.2f}")
    print(f"Best f1        : {best_f1:.4f}")
    print(f"Best threshold f1: {best_thresh_f1:.2f}")
    print(f"Confusion matrix (optimized for f1):\n{confusion_matrix_global}")

    return val_loss, best_thresh_acc, best_acc, best_thresh_f1, best_f1, confusion_matrix_global


N_EPOCHS = 250
LR_PATIENCE = 15
ES_PATIENCE = 20

CHECKPOINT_DIR = "/scratch/gpfs/MLISANTI/ra0438/cmae_checkpoints_freq"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SOURCE_TRIAL = 57 
BEST_PARAMS = {
    "embed_dim": 64,
    "dec_emb_frac": 1.0,
    "contrast_dim": 64,
    "mask_ratio": 0.41363797052507506,
    "dropout": 0.24308846333618442,
    "n_enc_heads": 4,
    "n_dec_frac": 0.25,
    "dim_feedforward": 256,
    "dim_feedforward_dec_frac": 0.25,
    "beta": 4.875295555739722,
    "gamma": 0.01205009026966113,
    "pos_weight_scalar": 1.530956225894752,
    "focal_gamma": 1.979087072188143,
    "n_enc_blocks": 8,
    "n_dec_block_frac": 1.0,
    "pretrain_frac": 0.6005005490559187,
    "lr": 0.0008125800628749078,
    "weight_decay": 0.00011947235079072891,
}


def build_model():
    p = BEST_PARAMS
    embed_dim = p["embed_dim"]
    dec_emb_dim = int(embed_dim * p["dec_emb_frac"])
    n_enc_heads = p["n_enc_heads"]
    n_dec_heads = max(1, int(n_enc_heads * p["n_dec_frac"]))
    dim_feedforward_enc = p["dim_feedforward"]
    dim_feedforward_dec = int(dim_feedforward_enc * p["dim_feedforward_dec_frac"])
    n_enc_blocks = p["n_enc_blocks"]
    n_dec_blocks = max(1, int(n_enc_blocks * p["n_dec_block_frac"]))

    model = FRBMaskedAutoencoder(
        seq_len=N_FREQ_CHANNELS,
        n_freq=TARGET_LENGTH,
        embed_dim=embed_dim,
        dec_embed_dim=dec_emb_dim,
        contrast_dim=p["contrast_dim"],
        mask_ratio=p["mask_ratio"],
        dropout=p["dropout"],
        n_enc_heads=n_enc_heads,
        n_dec_heads=n_dec_heads,
        dim_feedforward_enc=dim_feedforward_enc,
        dim_feedforward_dec=dim_feedforward_dec,
        n_enc_blocks=n_enc_blocks,
        n_dec_blocks=n_dec_blocks,
    ).to(device)
    return model


def run_fold(fold_idx, train_idx, val_idx):
    p = BEST_PARAMS
    lr = p["lr"]
    weight_decay = p["weight_decay"]
    beta = p["beta"]
    gamma = p["gamma"]
    pos_weight = p["pos_weight_scalar"]
    focal_param = p["focal_gamma"]
    pretrain_frac = p["pretrain_frac"]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    n_rep_train = int(dataset.labels[train_idx].sum())
    n_rep_val = int(dataset.labels[val_idx].sum())
    print(f"\n===== Fold {fold_idx + 1}/{N_SPLITS} =====")
    print(f"Train: {len(train_idx)} | repeaters: {n_rep_train} ({100 * n_rep_train / len(train_idx):.1f}%)")
    print(f"Val:   {len(val_idx)}   | repeaters: {n_rep_val}   ({100 * n_rep_val / len(val_idx):.1f}%)")

    model = build_model()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=LR_PATIENCE, min_lr=1e-6)

    n_pretrain_epochs = int(N_EPOCHS * pretrain_frac)
    for epoch in range(n_pretrain_epochs):
        pretrain_one_epoch(model, train_loader, optimizer, device, beta=beta, gamma=gamma, pos_weight_scalar=pos_weight)
        loss, recon_loss, con_loss = evaluate_pretrain(model, val_loader, device, beta=beta, gamma=gamma, pos_weight_scalar=pos_weight)
        scheduler.step(loss)
        print(f"Pretrain Epoch {epoch + 1}/{n_pretrain_epochs}: val_loss={loss:.4f}  val_recon={recon_loss:.4f}  val_con={con_loss:.4f}")

    for name, param in model.named_parameters():
        if any(name.startswith(pfx) for pfx in ("enc_proj", "enc_drop", "enc_pe", "enc_blocks", "enc_norm")):
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p_: p_.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=LR_PATIENCE, min_lr=1e-6)

    best_val_f1 = float("-inf")
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_confusion_matrix = None
    best_state = None
    best_thresh = 0.5
    best_thresh_acc = 0.5
    epochs_no_improve = 0

    n_finetune_epochs = int(N_EPOCHS * (1 - pretrain_frac))
    for epoch in range(n_finetune_epochs):
        finetune_one_epoch(model, train_loader, optimizer, device, alpha=1.0, pos_weight_scalar=pos_weight, focal_param=focal_param)

        val_loss, opt_thresh_acc, val_acc, opt_thresh, val_f1, cm = sweep_threshold(
            model, val_loader, device, alpha=1.0, beta=beta, gamma=gamma, pos_weight_scalar=pos_weight, focal_param=focal_param
        )

        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_confusion_matrix = cm
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_thresh = opt_thresh
            best_thresh_acc = opt_thresh_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= ES_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Fold {fold_idx + 1} result: val_loss={best_val_loss:.4f}, best_thresh(f1)={best_thresh:.3f}, "
          f"val_f1={best_val_f1:.3f}, val_acc={best_val_acc:.3f}, best_thresh_acc={best_thresh_acc:.3f}")
    print("  Confusion Matrix:")
    print(best_confusion_matrix)

    fold_path = os.path.join(CHECKPOINT_DIR, f"fold_{fold_idx}.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "params": BEST_PARAMS,
            "val_loss": best_val_loss,
            "val_f1": best_val_f1,
            "val_acc": best_val_acc,
            "best_thresh": best_thresh,
            "best_thresh_acc": best_thresh_acc,
            "fold": fold_idx,
            "confusion_matrix": best_confusion_matrix,
        },
        fold_path,
    )
    print(f"Fold {fold_idx + 1} checkpoint saved -> {fold_path}")

    return best_val_f1, best_val_acc, best_val_loss, best_confusion_matrix


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_f1s = []
fold_accs = []
fold_losses = []

indices = np.arange(n_total)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, dataset.labels)):
    val_f1, val_acc, val_loss, cm = run_fold(fold_idx, train_idx, val_idx)
    fold_f1s.append(val_f1)
    fold_accs.append(val_acc)
    fold_losses.append(val_loss)

fold_f1s = np.array(fold_f1s)
fold_accs = np.array(fold_accs)
fold_losses = np.array(fold_losses)

print("\n===== 5-Fold Cross-Validation Summary =====")
for i in range(N_SPLITS):
    print(f"Fold {i + 1}: F1={fold_f1s[i]:.4f}  Accuracy={fold_accs[i]:.4f}  Val Loss={fold_losses[i]:.4f}")

print(f"\nMean F1       : {fold_f1s.mean():.4f} +/- {fold_f1s.std():.4f}")
print(f"Mean Accuracy : {fold_accs.mean():.4f} +/- {fold_accs.std():.4f}")
print(f"Mean Val Loss : {fold_losses.mean():.4f} +/- {fold_losses.std():.4f}")

print(f"\nAll fold checkpoints saved to: {CHECKPOINT_DIR}")