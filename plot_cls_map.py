"""
Extract CLS tokens for every FRB in the dataset using the fold-2 model,
then run PCA on them and plot the result (repeater vs non-repeater).

Run this on the cluster where the checkpoint + hdf5/catalog files live.
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# ----------------------------------------------------------------------
# Dataset (identical to training script)
# ----------------------------------------------------------------------
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
        std[std == 0] = 1.0
        wfall = (wfall - wfall.mean(axis=1, keepdims=True)) / std

        peak = round(-extent[0] / self.dt)
        wfall = self._pad_or_crop(wfall, self.target_length, peak)
        tensor = torch.from_numpy(wfall)
        label = torch.tensor(int(key in self.repeater_set), dtype=torch.long)

        return tensor, label


# ----------------------------------------------------------------------
# Model pieces needed to reconstruct the architecture + load weights
# ----------------------------------------------------------------------
class WaterfallAugment(nn.Module):
    def __init__(self, time_mask_frac=0.15, freq_mask_frac=0.15, noise_std=0.05):
        super().__init__()
        self.time_mask_frac = time_mask_frac
        self.freq_mask_frac = freq_mask_frac
        self.noise_std = noise_std

    def forward(self, x):
        x = x.clone()
        B, n_freq, seq_len = x.shape

        t_mask = max(1, int(seq_len * self.time_mask_frac))
        for i in range(B):
            t0 = torch.randint(0, seq_len - t_mask, (1,)).item()
            x[i, :, t0:t0 + t_mask] = 0.0

        f_mask = max(1, int(n_freq * self.freq_mask_frac))
        for i in range(B):
            f0 = torch.randint(0, n_freq - f_mask, (1,)).item()
            x[i, f0:f0 + f_mask, :] = 0.0

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
    def __init__(self, seq_len, n_freq, embed_dim, dec_embed_dim, contrast_dim=32,
                 mask_ratio=0.25, dropout=0.1, n_enc_heads=4, n_dec_heads=2,
                 dim_feedforward_enc=128, dim_feedforward_dec=128,
                 n_enc_blocks=2, n_dec_blocks=2):
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
        noise = shared_noise if shared_noise is not None else torch.rand(B, T, device=x.device)
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
        x = x + self.enc_pe.pe[1: self.seq_len + 1]

        if mask_input:
            x_vis, mask, ids_restore, ids_keep = self.mask_input(x, shared_noise)
        else:
            x_vis = x
            mask = None
            ids_restore = None

        cls = (self.cls_token + self.enc_pe.pe[0]).expand(x_vis.size(0), -1, -1)
        x_vis = torch.cat([cls, x_vis], dim=1)

        for block in self.enc_blocks:
            x_vis = block(x_vis)
        x_vis = self.enc_norm(x_vis)

        return x_vis, mask, ids_restore


TARGET_LENGTH = 128
BATCH_SIZE = 64
NUM_WORKERS = 5
FOLD_IDX = 2  

N_SPLITS = 5
SEED = 42
EXCLUDE_FOLD = 4 

HDF5_PATH = "/scratch/gpfs/MLISANTI/ra0438/all_bursts.hdf5"
CATALOG_PATH = "/home/ra0438/chime-catalog-two-analysis/chimefrbcat2.csv"
CHECKPOINT_DIR = "/scratch/gpfs/MLISANTI/ra0438/cmae_checkpoints_5fold"
OUTPUT_DIR = "/scratch/gpfs/MLISANTI/ra0438/cmae_checkpoints_5fold/cls_pca"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_PARAMS = {
    "embed_dim": 256,
    "dec_emb_frac": 0.25,
    "contrast_dim": 16,
    "mask_ratio": 0.5391348878714695,
    "dropout": 0.37937057675681973,
    "n_enc_heads": 1,
    "n_dec_frac": 1.0,
    "dim_feedforward": 512,
    "dim_feedforward_dec_frac": 1.0,
    "beta": 4.948681956867076,
    "gamma": 0.06190300846556013,
    "pos_weight_scalar": 1.5981066428870525,
    "focal_gamma": -0.75780439499738,
    "n_enc_blocks": 6,
    "n_dec_block_frac": 0.25,
    "pretrain_frac": 0.6670128153150723,
    "lr": 0.0002312605096909245,
    "weight_decay": 0.008441184860364217,
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
        seq_len=TARGET_LENGTH,
        n_freq=256,
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


def main():
    full_dataset = CHIMEFRBDataset(HDF5_PATH, CATALOG_PATH, TARGET_LENGTH)
    n_total_full = len(full_dataset)
    n_rep_full = int(full_dataset.labels.sum())
    print(f"Full dataset: {n_total_full} | repeaters: {n_rep_full} ({100 * n_rep_full / n_total_full:.1f}%)")

    # Reproduce the exact same StratifiedKFold split used during training
    # (same N_SPLITS/seed), then drop the rows that fell in EXCLUDE_FOLD's
    # validation split so this run never touches fold 5's FRBs.
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    all_indices = np.arange(n_total_full)
    exclude_idx = None
    for fold_idx, (_, val_idx) in enumerate(skf.split(all_indices, full_dataset.labels)):
        if fold_idx == EXCLUDE_FOLD:
            exclude_idx = val_idx
            break

    keep_idx = np.setdiff1d(all_indices, exclude_idx)
    dataset = Subset(full_dataset, keep_idx)
    n_total = len(dataset)
    n_rep = int(full_dataset.labels[keep_idx].sum())
    print(f"Excluding fold {EXCLUDE_FOLD + 1} ({len(exclude_idx)} FRBs)")
    print(f"Remaining: {n_total} | repeaters: {n_rep} ({100 * n_rep / n_total:.1f}%)")

    model = build_model()
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold_{FOLD_IDX}.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    all_cls = []
    all_labels = []

    with torch.no_grad():
        for wfall, labels in loader:
            wfall = wfall.to(device)
            x_t = wfall.permute(0, 2, 1)  # [B, T, F]
            x_enc, _, _ = model.encoder(x_t, shared_noise=None, mask_input=False)
            cls_token = x_enc[:, 0, :]  # [B, embed_dim]
            all_cls.append(cls_token.cpu().numpy())
            all_labels.append(labels.numpy())

    all_cls = np.concatenate(all_cls, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Extracted CLS tokens: {all_cls.shape}")

    suffix = f"fold{FOLD_IDX}_excl_fold{EXCLUDE_FOLD + 1}"
    np.save(os.path.join(OUTPUT_DIR, f"cls_tokens_{suffix}.npy"), all_cls)
    np.save(os.path.join(OUTPUT_DIR, f"labels_{suffix}.npy"), all_labels)

    # ---- PCA ----
    pca = PCA(n_components=2)
    cls_2d = pca.fit_transform(all_cls)
    var_ratio = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    plt.scatter(cls_2d[all_labels == 0, 0], cls_2d[all_labels == 0, 1],
                s=8, alpha=0.5, label="Non-repeater", color="steelblue")
    plt.scatter(cls_2d[all_labels == 1, 0], cls_2d[all_labels == 1, 1],
                s=8, alpha=0.5, label="Repeater", color="crimson")
    plt.xlabel(f"PC1 ({var_ratio[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({var_ratio[1] * 100:.1f}% var)")
    plt.title(f"PCA of CLS tokens (Fold {FOLD_IDX} model, {n_total} FRBs, fold {EXCLUDE_FOLD + 1} excluded)")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, f"cls_pca_{suffix}.png")
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot -> {out_png}")


if __name__ == "__main__":
    main()