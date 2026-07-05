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
        std[std == 0] = 1.0
        wfall = (wfall - wfall.mean(axis=1, keepdims=True)) / std

        peak = round(-extent[0] / self.dt)
        wfall = self._pad_or_crop(wfall, self.target_length, peak)
        tensor = torch.from_numpy(wfall)
        label = torch.tensor(int(key in self.repeater_set), dtype=torch.long)

        return tensor, label


class WaterfallAugment(nn.Module):
    """Only needed because FRBMaskedAutoencoder.__init__ instantiates one --
    unused at eval time since we only call forward_finetune."""
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
            x[i, :, t0 : t0 + t_mask] = 0.0
        f_mask = max(1, int(n_freq * self.freq_mask_frac))
        for i in range(B):
            f0 = torch.randint(0, n_freq - f_mask, (1,)).item()
            x[i, f0 : f0 + f_mask, :] = 0.0
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

    def encoder(self, x, shared_noise=None, mask_input=True):
        x = self.enc_drop(self.enc_proj(x))
        x = x + self.enc_pe.pe[1 : self.seq_len + 1]

        x_vis = x
        cls = (self.cls_token + self.enc_pe.pe[0]).expand(x_vis.size(0), -1, -1)
        x_vis = torch.cat([cls, x_vis], dim=1)

        for block in self.enc_blocks:
            x_vis = block(x_vis)
        x_vis = self.enc_norm(x_vis)

        return x_vis, None, None

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


TARGET_LENGTH = 128
N_SPLITS = 5
BATCH_SIZE = 64
NUM_WORKERS = 5
SEED = 42

CHECKPOINT_DIR = "/scratch/gpfs/MLISANTI/ra0438/cmae_checkpoints_5fold"


def build_model_from_params(params):
    p = params
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


@torch.no_grad()
def get_probs_and_labels(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for wfall, labels in loader:
        wfall = wfall.to(device)
        cls_out = model.forward_finetune(wfall)
        probs = torch.sigmoid(cls_out).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def balanced_accuracy_from_cm(cm):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def sweep_balanced_accuracy(all_probs, all_labels):
    best_thresh, best_balacc = 0.5, -1.0
    best_cm = None

    for thresh in np.linspace(0.05, 0.95, 91):
        preds = (all_probs > thresh).astype(int)
        cm = confusion_matrix(all_labels, preds, labels=[0, 1])
        balacc = balanced_accuracy_from_cm(cm)
        if balacc > best_balacc:
            best_balacc = balacc
            best_thresh = thresh
            best_cm = cm

    best_preds = (all_probs > best_thresh).astype(int)
    acc = (best_preds == all_labels).mean()
    f1 = f1_score(all_labels, best_preds, pos_label=1)

    return best_thresh, best_balacc, acc, f1, best_cm


def main():
    dataset = CHIMEFRBDataset(
        hdf5_path="/scratch/gpfs/MLISANTI/ra0438/all_bursts.hdf5",
        catalog_path="/home/ra0438/chime-catalog-two-analysis/chimefrbcat2.csv",
        target_length=TARGET_LENGTH,
    )

    n_total = len(dataset)
    indices = np.arange(n_total)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(indices, dataset.labels))

    global_cm = np.zeros((2, 2), dtype=int)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold_{fold_idx}.pt")
        if not os.path.exists(ckpt_path):
            print(f"Fold {fold_idx}: checkpoint not found at {ckpt_path}, skipping.")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        params = ckpt["params"]

        model = build_model_from_params(params)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        val_ds = Subset(dataset, val_idx)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        all_probs, all_labels = get_probs_and_labels(model, val_loader, device)

        thresh, balacc, acc, f1, cm = sweep_balanced_accuracy(all_probs, all_labels)

        global_cm += cm

        n_rep_val = int(dataset.labels[val_idx].sum())
        print(f"\n===== Fold {fold_idx + 1}/{N_SPLITS} =====")
        print(f"Val: {len(val_idx)} | repeaters: {n_rep_val} ({100 * n_rep_val / len(val_idx):.1f}%)")
        print(f"Best threshold (balanced accuracy): {thresh:.2f}")
        print(f"Balanced accuracy : {balacc:.4f}")
        print(f"Accuracy          : {acc:.4f}")
        print(f"F1                : {f1:.4f}")
        print("Confusion matrix:")
        print(cm)

        fold_results.append(
            {
                "fold": fold_idx,
                "threshold": thresh,
                "balanced_accuracy": balacc,
                "accuracy": acc,
                "f1": f1,
                "confusion_matrix": cm,
            }
        )

    balaccs = np.array([r["balanced_accuracy"] for r in fold_results])
    accs = np.array([r["accuracy"] for r in fold_results])
    f1s = np.array([r["f1"] for r in fold_results])

    print("\n===== Per-Fold Summary =====")
    for r in fold_results:
        print(f"Fold {r['fold'] + 1}: BalAcc={r['balanced_accuracy']:.4f}  "
              f"Accuracy={r['accuracy']:.4f}  F1={r['f1']:.4f}  thresh={r['threshold']:.2f}")

    print(f"\nMean Balanced Accuracy : {balaccs.mean():.4f} +/- {balaccs.std():.4f}")
    print(f"Mean Accuracy          : {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"Mean F1                : {f1s.mean():.4f} +/- {f1s.std():.4f}")

    # Pooled/global confusion matrix: every sample in the dataset was
    # validated on exactly once across the 5 folds (each using its own
    # balanced-accuracy-optimal threshold), so summing gives a single
    # confusion matrix covering the whole dataset.
    global_tn, global_fp, global_fn, global_tp = global_cm[0, 0], global_cm[0, 1], global_cm[1, 0], global_cm[1, 1]
    global_balacc = balanced_accuracy_from_cm(global_cm)
    global_acc = (global_tn + global_tp) / global_cm.sum()
    global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    global_f1 = (2 * global_precision * global_recall / (global_precision + global_recall)
                 if (global_precision + global_recall) > 0 else 0.0)

    print("\n===== Global (Pooled Out-of-Fold) Confusion Matrix =====")
    print(global_cm)
    print(f"Global Balanced Accuracy : {global_balacc:.4f}")
    print(f"Global Accuracy          : {global_acc:.4f}")
    print(f"Global Precision         : {global_precision:.4f}")
    print(f"Global Recall            : {global_recall:.4f}")
    print(f"Global F1                : {global_f1:.4f}")


if __name__ == "__main__":
    main()