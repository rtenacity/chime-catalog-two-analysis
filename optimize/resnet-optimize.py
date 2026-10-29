import h5py
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import optuna
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()


# ----------------------------------------------------------------------------
# Data (unchanged)
# ----------------------------------------------------------------------------
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
        tensor = torch.from_numpy(np.ascontiguousarray(wfall))
        label = torch.tensor(int(key in self.repeater_set), dtype=torch.long)

        return tensor, label


def make_dataloader(hdf5_path: str, catalog_path: str, target_length: int, batch_size: int = 32,
                    shuffle: bool = True, num_workers: int = 0,
                    train_frac: float = 0.66, holdout_frac: float = 0.15, seed: int = 42):
    dataset = CHIMEFRBDataset(hdf5_path, catalog_path, target_length)
    n_total = len(dataset)

    labels = dataset.labels
    indices = list(range(n_total))

    trainval_idx, holdout_idx = train_test_split(
        indices, test_size=holdout_frac, stratify=labels, random_state=seed
    )

    trainval_labels = [labels[i] for i in trainval_idx]
    val_frac_of_trainval = (1 - train_frac - holdout_frac) / (1 - holdout_frac)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_frac_of_trainval, stratify=trainval_labels, random_state=seed
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    holdout_ds = Subset(dataset, holdout_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    holdout_loader = DataLoader(holdout_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    def _rep_stats(name, idx):
        n_rep = sum(labels[i] for i in idx)
        print(f"{name}: {len(idx)} | repeaters: {n_rep} ({100*n_rep/len(idx):.1f}%)")

    _rep_stats("Train  ", train_idx)
    _rep_stats("Val    ", val_idx)
    _rep_stats("Holdout", holdout_idx)

    return train_loader, val_loader, holdout_loader


TARGET_LENGTH = 128       # number of time samples kept per burst window
N_FREQ_CHANNELS = 256     # number of frequency channels in the waterfall


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
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
        bce = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight, reduction="none"
        )
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al. 2020). features: [B, n_views, D], L2-normalised."""
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        dev = features.device
        if features.dim() < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if features.dim() > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=dev)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(dev)
        else:
            mask = mask.float().to(dev)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature, anchor_count = features[:, 0], 1
        elif self.contrast_mode == "all":
            anchor_feature, anchor_count = contrast_feature, contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        logits = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count, device=dev).view(-1, 1), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class WaterfallAugment(nn.Module):
    """Second view for SupCon: random time crop/shift, time mask, freq mask, Gaussian noise."""
    def __init__(self, time_mask_frac=0.15, freq_mask_frac=0.15, noise_std=0.05, crop_frac_range=(0.7, 1.0)):
        super().__init__()
        self.time_mask_frac = time_mask_frac
        self.freq_mask_frac = freq_mask_frac
        self.noise_std = noise_std
        self.crop_frac_range = crop_frac_range

    def _random_time_crop(self, x):
        # pads with zeroes (ref https://arxiv.org/pdf/2103.01929)
        B, n_freq, seq_len = x.shape
        out = torch.zeros_like(x)
        for i in range(B):
            frac = float(torch.empty(1).uniform_(*self.crop_frac_range))
            crop_len = min(seq_len, max(1, int(seq_len * frac)))
            src_start = int(torch.randint(0, seq_len - crop_len + 1, (1,)))
            dst_start = int(torch.randint(0, seq_len - crop_len + 1, (1,)))
            out[i, :, dst_start:dst_start + crop_len] = x[i, :, src_start:src_start + crop_len]
        return out

    def forward(self, x):
        x = self._random_time_crop(x.clone())
        B, n_freq, seq_len = x.shape

        t_mask = max(1, int(seq_len * self.time_mask_frac))
        f_mask = max(1, int(n_freq * self.freq_mask_frac))
        for i in range(B):
            t0 = torch.randint(0, seq_len - t_mask, (1,)).item()
            x[i, :, t0:t0 + t_mask] = 0.0
            f0 = torch.randint(0, n_freq - f_mask, (1,)).item()
            x[i, f0:f0 + f_mask, :] = 0.0

        return x + torch.randn_like(x) * self.noise_std


def compute_loss(logits, recon, wfall, labels, emb_views=None, mask=None, alpha=1.0, beta=1.0, gamma=0.1,
                 pos_weight=None, focal_param=0.0, supcon_fn=None):
    """Joint objective: alpha * classification + beta * reconstruction + gamma * SupCon.

    logits    : [B]          recon / wfall : [B, F, T]
    emb_views : [B, n_views, D] L2-normalised embeddings, or None (con loss = 0).
    mask      : [B, F, T] with 1 = pixel was hidden from the encoder, or None.
                If given, recon loss is computed on hidden pixels only (MAE-style);
                otherwise on every pixel (plain autoencoder).
    """
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=logits.device) if pos_weight else None
    cls_loss = FocalLoss(gamma=focal_param, pos_weight=pw)(logits, labels.float())

    sq = (recon - wfall) ** 2
    if mask is not None and mask.sum() > 0:
        recon_loss = (sq * mask).sum() / mask.sum()
    else:
        recon_loss = sq.mean()

    if emb_views is not None and supcon_fn is not None:
        con_loss = supcon_fn(emb_views, labels=labels)
    else:
        con_loss = torch.zeros((), device=logits.device)

    total = alpha * cls_loss + beta * recon_loss + gamma * con_loss
    return total, cls_loss, recon_loss, con_loss


# ----------------------------------------------------------------------------
# Model: ResNet conv encoder -> bottleneck -> (ResNet conv decoder, classifier head)
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Pre-activation basic residual block (BN-ReLU-Conv x2) with optional stride."""
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(self.drop(F.relu(self.bn2(out))))
        return out + sc


class UpBlock(nn.Module):
    """2x upsample (transposed conv) followed by residual blocks."""
    def __init__(self, in_ch, out_ch, n_blocks, dropout=0.0):
        super().__init__()
        self.up = nn.Sequential(nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False))
        self.blocks = nn.Sequential(*[ResBlock(out_ch, out_ch, dropout=dropout) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(self.up(x))


class FRBResNetAE(nn.Module):
    """
    Input    : [B, F, T] waterfall (treated as a 1-channel image).
    Encoder  : stem conv -> n_stages ResNet stages, each halving F and T and doubling channels
               (capped at max_channels) -> latent map z [B, L, F', T'].
    Decoder  : mirror of the encoder with transposed-conv upsampling; no U-Net skips, so
               everything must pass through the bottleneck.
    Embedding: global mean pool of z -> MLP -> vector e [B, embed_dim]. This one vector is
               (a) L2-normalised for SupCon, (b) the thing to visualise (UMAP/t-SNE), and
               (c) the input to the classifier head.
    Head     : e -> MLP -> 1 logit.
    Optional patch masking on the input (MAE-style) during training.
    """
    def __init__(self, n_freq, n_time, base_channels=32, n_stages=3, enc_blocks=2, dec_blocks=1,
                 max_channels=256, latent_channels=None, embed_dim=64, dropout=0.1, head_hidden=128,
                 mask_ratio=0.0, mask_patch_f=16, mask_patch_t=8):
        super().__init__()
        down = 2 ** n_stages
        assert n_freq % down == 0 and n_time % down == 0, "n_freq/n_time must be divisible by 2**n_stages"
        assert n_freq % mask_patch_f == 0 and n_time % mask_patch_t == 0

        self.n_freq, self.n_time = n_freq, n_time
        self.mask_ratio = mask_ratio
        self.mask_patch_f, self.mask_patch_t = mask_patch_f, mask_patch_t

        chs = [min(base_channels * 2 ** i, max_channels) for i in range(n_stages + 1)]
        latent_channels = latent_channels or chs[-1]

        # ---- encoder ----
        self.stem = nn.Conv2d(1, chs[0], 3, padding=1, bias=False)
        enc = []
        for i in range(n_stages):
            stage = [ResBlock(chs[i], chs[i + 1], stride=2, dropout=dropout)]
            stage += [ResBlock(chs[i + 1], chs[i + 1], dropout=dropout) for _ in range(enc_blocks - 1)]
            enc.append(nn.Sequential(*stage))
        self.enc_stages = nn.Sequential(*enc)
        self.to_latent = nn.Sequential(nn.BatchNorm2d(chs[-1]), nn.ReLU(inplace=True),
                                       nn.Conv2d(chs[-1], latent_channels, 1))

        # ---- decoder ----
        self.from_latent = nn.Conv2d(latent_channels, chs[-1], 1)
        dec = [UpBlock(chs[i + 1], chs[i], dec_blocks, dropout=dropout) for i in reversed(range(n_stages))]
        self.dec_stages = nn.Sequential(*dec)
        self.dec_out = nn.Sequential(nn.BatchNorm2d(chs[0]), nn.ReLU(inplace=True),
                                     nn.Conv2d(chs[0], 1, 3, padding=1))

        # ---- embedding (mean-pooled latent -> vector) ----
        self.embed_head = nn.Sequential(nn.Linear(latent_channels, latent_channels),
                                        nn.LayerNorm(latent_channels),
                                        nn.ReLU(),
                                        nn.Linear(latent_channels, embed_dim))

        # ---- classifier head (on the embedding) ----
        self.cls_head = nn.Sequential(nn.Linear(embed_dim, head_hidden),
                                      nn.LayerNorm(head_hidden),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(head_hidden, head_hidden // 2),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(head_hidden // 2, 1))

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def random_patch_mask(self, x):
        """Returns pixel mask [B, F, T] with 1 = hidden, hiding exactly mask_ratio of the patches."""
        B = x.size(0)
        gf, gt = self.n_freq // self.mask_patch_f, self.n_time // self.mask_patch_t
        n = gf * gt
        n_mask = int(round(n * self.mask_ratio))
        noise = torch.rand(B, n, device=x.device)
        ids = torch.argsort(noise, dim=1)[:, :n_mask]
        m = torch.zeros(B, n, device=x.device)
        m.scatter_(1, ids, 1.0)
        m = m.view(B, gf, gt)
        m = m.repeat_interleave(self.mask_patch_f, dim=1).repeat_interleave(self.mask_patch_t, dim=2)
        return m

    def encode(self, x):
        h = self.stem(x.unsqueeze(1))                # [B, C0, F, T]
        h = self.enc_stages(h)                       # [B, C_n, F/2^n, T/2^n]
        return self.to_latent(h)                     # [B, L, F/2^n, T/2^n]

    def decode(self, z):
        h = self.dec_stages(self.from_latent(z))
        return self.dec_out(h).squeeze(1)            # [B, F, T]

    def embed(self, z):
        return self.embed_head(z.mean(dim=(2, 3)))   # [B, embed_dim]

    def classify(self, e):
        return self.cls_head(e).squeeze(-1)          # [B]

    @torch.no_grad()
    def get_embedding(self, x, normalize=True):
        """Clean-input embedding for visualisation. Call model.eval() first."""
        e = self.embed(self.encode(x))
        return F.normalize(e, dim=1) if normalize else e

    def forward(self, x, apply_mask=None):
        """Single pass on view 1: returns (logits, recon, mask, emb). Masking defaults to train mode only.
        emb is the raw (un-normalised) embedding vector."""
        if apply_mask is None:
            apply_mask = self.training
        mask = None
        x_in = x
        if apply_mask and self.mask_ratio > 0:
            mask = self.random_patch_mask(x)
            x_in = x * (1 - mask)
        z = self.encode(x_in)
        e = self.embed(z)
        return self.classify(e), self.decode(z), mask, e

    def forward_contrastive(self, x, x_aug, apply_mask=None):
        """View 1 = (optionally masked) x -> logits, recon, embedding.
        View 2 = augmented x_aug -> embedding only.
        Returns (logits, recon, mask, emb_views [B, 2, D] L2-normalised)."""
        logits, recon, mask, e1 = self(x, apply_mask=apply_mask)
        e2 = self.embed(self.encode(x_aug))
        views = torch.stack([F.normalize(e1, dim=1), F.normalize(e2, dim=1)], dim=1)
        return logits, recon, mask, views


# ----------------------------------------------------------------------------
# Train / eval (single joint phase)
# ----------------------------------------------------------------------------
augment_fn = WaterfallAugment(time_mask_frac=0.15, freq_mask_frac=0.15, noise_std=0.05)


def train_one_epoch(model, loader, optimizer, device, alpha, beta, gamma, pos_weight, focal_param, supcon_fn):
    model.train()
    run = {"loss": 0.0, "cls": 0.0, "recon": 0.0, "con": 0.0}
    correct = total = 0

    for wfall, labels in loader:
        wfall, labels = wfall.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        logits, recon, mask, views = model.forward_contrastive(wfall, augment_fn(wfall))
        loss, cls_loss, recon_loss, con_loss = compute_loss(
            logits, recon, wfall, labels, emb_views=views, mask=mask, alpha=alpha, beta=beta, gamma=gamma,
            pos_weight=pos_weight, focal_param=focal_param, supcon_fn=supcon_fn)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        run["loss"] += loss.item()
        run["cls"] += cls_loss.item()
        run["recon"] += recon_loss.item()
        run["con"] += con_loss.item()
        correct += ((torch.sigmoid(logits) > 0.5).long() == labels).sum().item()
        total += labels.size(0)

    n = len(loader)
    print(f"  train loss={run['loss']/n:.4f}  cls={run['cls']/n:.4f}  recon={run['recon']/n:.4f}  "
          f"con={run['con']/n:.4f}  acc={correct/total:.3f}")


@torch.no_grad()
def evaluate(model, loader, device, alpha, beta, gamma, pos_weight, focal_param, supcon_fn, verbose=True):
    """Clean (unmasked) view 1 for classification, full-image recon and embeddings; augmented view 2
    for the SupCon term. Sweeps the decision threshold."""
    model.eval()
    all_probs, all_labels, all_emb = [], [], []
    run = {"loss": 0.0, "cls": 0.0, "recon": 0.0, "con": 0.0}

    for wfall, labels in loader:
        wfall, labels = wfall.to(device), labels.to(device)
        logits, recon, _, views = model.forward_contrastive(wfall, augment_fn(wfall), apply_mask=False)
        loss, cls_loss, recon_loss, con_loss = compute_loss(
            logits, recon, wfall, labels, emb_views=views, mask=None, alpha=alpha, beta=beta, gamma=gamma,
            pos_weight=pos_weight, focal_param=focal_param, supcon_fn=supcon_fn)
        run["loss"] += loss.item()
        run["cls"] += cls_loss.item()
        run["recon"] += recon_loss.item()
        run["con"] += con_loss.item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_emb.append(views[:, 0].cpu().numpy())   # clean, L2-normalised embedding

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_emb = np.concatenate(all_emb)
    n = len(loader)

    threshs = np.linspace(0.05, 0.95, 91)
    accs = [((all_probs > t).astype(int) == all_labels).mean() for t in threshs]
    f1s = [f1_score(all_labels, (all_probs > t).astype(int), pos_label=1, zero_division=0) for t in threshs]
    i_acc, i_f1 = int(np.argmax(accs)), int(np.argmax(f1s))

    best_thresh_acc, best_acc = threshs[i_acc], accs[i_acc]
    best_thresh_f1, best_f1 = threshs[i_f1], f1s[i_f1]
    cm = confusion_matrix(all_labels, (all_probs > best_thresh_f1).astype(int), labels=[0, 1])

    out = {"loss": run["loss"] / n, "cls_loss": run["cls"] / n, "recon_loss": run["recon"] / n,
           "con_loss": run["con"] / n,
           "acc": best_acc, "thresh_acc": best_thresh_acc, "f1": best_f1, "thresh_f1": best_thresh_f1,
           "cm": cm, "probs": all_probs, "labels": all_labels, "emb": all_emb}

    if verbose:
        print(f"  val loss={out['loss']:.4f}  cls={out['cls_loss']:.4f}  recon={out['recon_loss']:.4f}  "
              f"con={out['con_loss']:.4f}")
        print(f"  best acc={best_acc:.4f} @ {best_thresh_acc:.2f}   best f1={best_f1:.4f} @ {best_thresh_f1:.2f}")
        print(f"  confusion matrix (f1-optimal):\n{cm}")
    return out


@torch.no_grad()
def evaluate_at_threshold(model, loader, device, thresh):
    """Holdout evaluation at a fixed threshold chosen on validation (no re-sweeping)."""
    model.eval()
    probs, labels, embs = [], [], []
    for wfall, y in loader:
        logits, _, _, e = model(wfall.to(device), apply_mask=False)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        embs.append(F.normalize(e, dim=1).cpu().numpy())
        labels.append(y.numpy())
    probs, labels, embs = np.concatenate(probs), np.concatenate(labels), np.concatenate(embs)
    preds = (probs > thresh).astype(int)
    return {"acc": (preds == labels).mean(),
            "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
            "cm": confusion_matrix(labels, preds, labels=[0, 1]),
            "probs": probs, "labels": labels, "emb": embs}


# ----------------------------------------------------------------------------
# Optuna
# ----------------------------------------------------------------------------
N_EPOCHS = 250
CHECKPOINT_DIR = "/scratch/gpfs/MLISANTI/ra0438/resnet_ae_checkpoints_holdout"
STUDY_DB = "sqlite:////scratch/gpfs/MLISANTI/ra0438/resnet_ae_study_holdout.db"

best_overall = {"f1": float("-inf")}


def build_model_from_params(p):
    mask_patch_f, mask_patch_t = map(int, p["mask_patch"].split("x"))
    return FRBResNetAE(n_freq=N_FREQ_CHANNELS, n_time=TARGET_LENGTH,
                       base_channels=p["base_channels"], n_stages=p["n_stages"],
                       enc_blocks=p["enc_blocks"], dec_blocks=p["dec_blocks"],
                       max_channels=p["max_channels"], latent_channels=p["latent_channels"],
                       embed_dim=p["embed_dim"], dropout=p["dropout"], head_hidden=p["head_hidden"],
                       mask_ratio=p["mask_ratio"], mask_patch_f=mask_patch_f, mask_patch_t=mask_patch_t)


def objective(trial):
    set_seed()
    lr_patience = 15
    es_patience = 20

    p = {
        "base_channels": trial.suggest_categorical("base_channels", [8, 16, 32, 64]),
        "n_stages": trial.suggest_categorical("n_stages", [2, 3, 4, 5]),
        "enc_blocks": trial.suggest_categorical("enc_blocks", [1, 2, 3]),
        "dec_blocks": trial.suggest_categorical("dec_blocks", [1, 2]),
        "max_channels": trial.suggest_categorical("max_channels", [128, 256, 512]),
        "latent_channels": trial.suggest_categorical("latent_channels", [32, 64, 128, 256]),
        "embed_dim": trial.suggest_categorical("embed_dim", [16, 32, 64, 128]),
        "head_hidden": trial.suggest_categorical("head_hidden", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "mask_ratio": trial.suggest_float("mask_ratio", 0.0, 0.6),
        "mask_patch": trial.suggest_categorical("mask_patch", ["8x4", "16x8", "32x8", "16x16", "32x16"]),
    }
    alpha = 1.0
    beta = trial.suggest_float("beta", 0.01, 10.0, log=True)
    gamma = trial.suggest_float("gamma", 0.01, 1.0, log=True)
    temperature = trial.suggest_categorical("temperature", [0.05, 0.07, 0.1, 0.2])
    supcon_fn = SupConLoss(temperature=temperature, contrast_mode="all", base_temperature=temperature)
    pos_weight = trial.suggest_float("pos_weight_scalar", 1.0, 5.0)
    focal_param = trial.suggest_float("focal_gamma", 0.0, 3.0)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = build_model_from_params(p).to(device)
    n_params = sum(q.numel() for q in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=lr_patience, min_lr=1e-6)

    print(f"Trial {trial.number} ({n_params/1e6:.2f}M params): {trial.params}")

    best = {"f1": float("-inf")}
    best_state = None
    epochs_no_improve = 0

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        train_one_epoch(model, train_loader, optimizer, device, alpha, beta, gamma, pos_weight, focal_param, supcon_fn)
        val = evaluate(model, val_loader, device, alpha, beta, gamma, pos_weight, focal_param, supcon_fn)

        scheduler.step(val["loss"])
        trial.report(val["f1"], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val["f1"] > best["f1"]:
            best = {k: val[k] for k in ("f1", "acc", "thresh_f1", "thresh_acc", "loss", "cls_loss",
                                        "recon_loss", "con_loss", "cm")}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= es_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Trial {trial.number}: val_loss={best['loss']:.4f} (cls={best['cls_loss']:.4f}, "
          f"recon={best['recon_loss']:.4f}, con={best['con_loss']:.4f}), "
          f"val_f1={best['f1']:.3f} @ {best['thresh_f1']:.2f}, val_acc={best['acc']:.3f} @ {best['thresh_acc']:.2f}")
    print(f"  Confusion Matrix:\n{best['cm']}")

    ckpt = {"model_state_dict": best_state, "params": trial.params, "model_params": p,
            "val_loss": best["loss"], "val_cls_loss": best["cls_loss"], "val_recon_loss": best["recon_loss"],
            "val_con_loss": best["con_loss"],
            "val_f1": best["f1"], "best_thresh": best["thresh_f1"],
            "val_acc": best["acc"], "best_thresh_acc": best["thresh_acc"],
            "trial_number": trial.number, "confusion_matrix": best["cm"]}
    trial_path = os.path.join(CHECKPOINT_DIR, f"trial_{trial.number}.pt")
    torch.save(ckpt, trial_path)
    print(f"Trial {trial.number} checkpoint saved -> {trial_path}")

    if best["f1"] > best_overall["f1"]:
        best_overall["f1"] = best["f1"]
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save(ckpt, best_path)
        print(f"New best model saved -> {best_path}  (val_f1={best['f1']:.3f}, trial={trial.number})")

    return best["f1"]



os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_loader, val_loader, holdout_loader = make_dataloader(
    hdf5_path="/scratch/gpfs/MLISANTI/ra0438/all_bursts.hdf5",
    catalog_path="/home/ra0438/chime-catalog-two-analysis/chimefrbcat2.csv",
    train_frac=0.7, holdout_frac=0.15, target_length=TARGET_LENGTH, batch_size=64, num_workers=5)

for wfall_batch, label_batch in train_loader:
    print(f"Batch shape : {wfall_batch.shape}")
    print(f"Labels      : {label_batch}")
    break

study = optuna.create_study(
    study_name="resnet_ae_joint",
    storage=STUDY_DB,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15),
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)

print("\nBest trial:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")
print(f"\nAll checkpoints saved to: {CHECKPOINT_DIR}")
print(f"Best overall val_f1 (this run): {best_overall['f1']:.3f}")
