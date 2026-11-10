"""
FAME: Frequency-Aware Multi-Scale Interest Encoding with Global Graph Modeling
for Diversity-Oriented Sequential Recommendation
Real-dataset version.

Dependencies:
    pip install torch numpy scipy PyWavelets EMD-signal tqdm

Dataset layout (Windows example):
    D:/datasets/ratings.dat          (MovieLens-1M)
    D:/datasets/Beauty_5.json.gz     (Amazon Beauty 5-core)

Usage:
    python fame.py --dataset ml-1m  --data-root D:/datasets
    python fame.py --dataset beauty --data-root D:/datasets
    python fame.py --dataset ml-1m  --data-root D:/datasets --epochs 30
"""

import argparse
import gzip
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pywt
from PyEMD import EMD

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback
    def tqdm(x, **kwargs):
        return x


# =====================================================================
# 0. Configuration
# =====================================================================
class Config:
    seq_len = 50
    embed_dim = 64
    hidden_dim = 64
    n_layers_lstm = 1
    dropout = 0.2
    wavelet = "db4"
    wavelet_level = 3
    emd_max_imf = 6
    graph_layers = 2
    beta = 0.1
    delta = 0.5
    lr = 1e-3
    batch_size = 256
    n_neg = 100
    pop_exp = 0.75
    n_epochs = 20
    patience = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"


# MovieLens / Amazon ratings (1-5) → interest strength
RATING_TO_STRENGTH = {1: 0.2, 2: 0.4, 3: 0.7, 4: 0.85, 5: 1.0}


def rating_to_strength(r):
    return RATING_TO_STRENGTH.get(int(round(r)), 0.5)


# =====================================================================
# 1. Data loading (real datasets)
# =====================================================================
def _iterative_5core_filter(sequences):
    """Iteratively drop users / items with < 5 interactions."""
    sequences = {u: list(s) for u, s in sequences.items()}
    while True:
        user_counts = {u: len(s) for u, s in sequences.items()}
        item_counts = defaultdict(int)
        for s in sequences.values():
            for i in s:
                item_counts[i] += 1

        users_drop = {u for u, c in user_counts.items() if c < 5}
        items_drop = {i for i, c in item_counts.items() if c < 5}
        if not users_drop and not items_drop:
            break

        new_seqs = {}
        for u, s in sequences.items():
            if u in users_drop:
                continue
            f = [i for i in s if i not in items_drop]
            if len(f) >= 5:
                new_seqs[u] = f
        sequences = new_seqs
    return sequences


def _remap_ids(sequences, strengths):
    """User IDs → 0..N-1; item IDs → 1..M (0 reserved for padding)."""
    user_ids = sorted(sequences.keys())
    all_items = sorted({i for s in sequences.values() for i in s})
    u2i = {u: i for i, u in enumerate(user_ids)}
    i2i = {item: i + 1 for i, item in enumerate(all_items)}

    new_seqs, new_str = {}, {}
    for u in user_ids:
        new_seqs[u2i[u]] = [i2i[it] for it in sequences[u]]
        new_str[u2i[u]] = strengths[u]
    return new_seqs, new_str, len(user_ids), len(all_items)


def load_movielens_1m(data_root):
    """Load MovieLens-1M ratings.dat. Format: UserID::MovieID::Rating::Timestamp."""
    candidates = [
        os.path.join(data_root, "ratings.dat"),
        os.path.join(data_root, "ml-1m", "ratings.dat"),
        os.path.join(data_root, "movielens", "ml-1m", "ratings.dat"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"ratings.dat not found. Expected at {candidates[0]} "
            f"or {candidates[1]}. Download from https://grouplens.org/datasets/movielens/1m/"
        )
    print(f"[data] MovieLens-1M ← {path}")

    user2records = defaultdict(list)
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 4:
                continue
            uid, iid, rating, ts = parts
            user2records[int(uid)].append((int(ts), int(iid), int(rating)))

    sequences, strengths = {}, {}
    for uid, recs in user2records.items():
        recs.sort(key=lambda x: x[0])
        sequences[uid] = [r[1] for r in recs]
        strengths[uid] = [rating_to_strength(r[2]) for r in recs]
    return sequences, strengths


def load_amazon_beauty(data_root):
    """Load Amazon Beauty 5-core JSON (one record per line)."""
    candidates = [
        os.path.join(data_root, "Beauty_5.json.gz"),
        os.path.join(data_root, "Beauty_5.json"),
        os.path.join(data_root, "amazon_beauty", "Beauty_5.json.gz"),
        os.path.join(data_root, "amazon_beauty", "Beauty_5.json"),
        os.path.join(data_root, "beauty", "Beauty_5.json.gz"),
        os.path.join(data_root, "beauty", "Beauty_5.json"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"Beauty_5.json[.gz] not found. Expected one of: {candidates}. "
            f"Download from https://jmcauley.ucsd.edu/data/amazon/"
        )
    print(f"[data] Amazon Beauty ← {path}")

    user2records = defaultdict(list)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec["reviewerID"]
            iid = rec["asin"]
            ts = float(rec.get("unixReviewTime", 0.0))
            rating = float(rec.get("overall", 5.0))
            user2records[uid].append((ts, iid, rating))

    # string IDs → temporary ints
    item2int = {}
    sequences, strengths = {}, {}
    for uid, recs in user2records.items():
        recs.sort(key=lambda x: x[0])
        seq = []
        for _, iid, _ in recs:
            if iid not in item2int:
                item2int[iid] = len(item2int) + 1
            seq.append(item2int[iid])
        sequences[uid] = seq
        strengths[uid] = [rating_to_strength(r[2]) for r in recs]

    # string user IDs → ints (kept as strings here, remapped in _remap_ids)
    uid2int = {uid: i for i, uid in enumerate(sequences.keys())}
    new_seqs = {uid2int[u]: s for u, s in sequences.items()}
    new_str = {uid2int[u]: s for u, s in strengths.items()}
    return new_seqs, new_str


def prepare_dataset(name, data_root):
    """Load, 5-core filter, and remap a real dataset."""
    if name == "ml-1m":
        sequences, strengths = load_movielens_1m(data_root)
    elif name == "beauty":
        sequences, strengths = load_amazon_beauty(data_root)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    n_users_raw = len(sequences)
    n_items_raw = len({i for s in sequences.values() for i in s})
    print(f"[data] raw: {n_users_raw} users, {n_items_raw} items, "
          f"{sum(len(s) for s in sequences.values())} interactions")

    # 5-core filter (also filters strengths in lockstep)
    filtered_seqs = _iterative_5core_filter(sequences)
    filtered_str = {u: strengths[u][: len(filtered_seqs[u])] if False else None
                    for u in filtered_seqs}
    # Correct: align by (user → retained items). Since 5-core only drops items
    # from the tail/head, we rebuild strengths using the same ordering logic.
    kept_items = {u: set(s) for u, s in filtered_seqs.items()}
    filtered_str = {}
    for u, items in filtered_seqs.items():
        orig_items, orig_str = sequences[u], strengths[u]
        filtered_str[u] = [s for it, s in zip(orig_items, orig_str)
                           if it in kept_items[u]]

    n_users = len(filtered_seqs)
    n_items = len({i for s in filtered_seqs.values() for i in s})
    n_inter = sum(len(s) for s in filtered_seqs.values())
    density = n_inter / (n_users * n_items) if n_users and n_items else 0.0
    print(f"[data] after 5-core: {n_users} users, {n_items} items, "
          f"{n_inter} interactions, density={density:.4%}")

    sequences, strengths, n_users, n_items = _remap_ids(filtered_seqs, filtered_str)
    print(f"[data] remapped: user∈[0,{n_users-1}], item∈[1,{n_items}] (0=padding)")
    return sequences, strengths, n_users, n_items


# =====================================================================
# 2. Signal processing: interest-intensity signal, EMD, wavelet
# =====================================================================
class SignalProcessor:
    """Turns a strength sequence into three denoised band signals."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.emd = EMD()

    @staticmethod
    def _resample(signal, L):
        T = len(signal)
        if T == 0:
            return np.zeros(L, dtype=np.float32)
        if T == L:
            return np.asarray(signal, dtype=np.float32).copy()
        x_old = np.linspace(0.0, 1.0, T)
        x_new = np.linspace(0.0, 1.0, L)
        return np.interp(x_new, x_old, signal).astype(np.float32)

    @staticmethod
    def _normalize(signal):
        mu = float(signal.mean())
        sd = float(signal.std())
        if sd < 1e-8:
            return (signal - mu).astype(np.float32)
        return ((signal - mu) / sd).astype(np.float32)

    def _emd(self, signal):
        try:
            imfs = self.emd(signal)
        except Exception:
            return np.zeros((0, len(signal)), dtype=np.float32), signal.copy()
        if imfs is None or len(imfs) == 0:
            return np.zeros((0, len(signal)), dtype=np.float32), signal.copy()
        imfs = np.atleast_2d(imfs)
        if imfs.shape[0] > self.cfg.emd_max_imf:
            imfs = imfs[: self.cfg.emd_max_imf]
        residual = signal - imfs.sum(axis=0)
        return imfs.astype(np.float32), residual.astype(np.float32)

    @staticmethod
    def _mean_freq(imf):
        s = np.sign(imf)
        s[s == 0] = 1.0
        crossings = int(np.sum(np.abs(np.diff(s)) > 0))
        return crossings / max(len(imf) - 1, 1)

    def _group_bands(self, imfs, residual):
        L = len(residual)
        if len(imfs) == 0:
            return np.zeros(L, np.float32), np.zeros(L, np.float32), np.zeros(L, np.float32)
        if len(imfs) < 3:
            return residual.copy(), imfs.sum(axis=0).astype(np.float32), np.zeros(L, np.float32)

        freqs = np.array([self._mean_freq(c) for c in imfs])
        tau_h = np.percentile(freqs, 66)
        tau_l = np.percentile(freqs, 33)
        high = freqs >= tau_h
        mid = (freqs >= tau_l) & (freqs < tau_h)
        low = freqs < tau_l

        x_high = imfs[high].sum(axis=0) if high.any() else np.zeros(L, np.float32)
        x_mid = imfs[mid].sum(axis=0) if mid.any() else np.zeros(L, np.float32)
        x_low = imfs[low].sum(axis=0) if low.any() else np.zeros(L, np.float32)
        x_low = x_low + residual
        return x_low.astype(np.float32), x_mid.astype(np.float32), x_high.astype(np.float32)

    def _wavelet_denoise(self, signal):
        L = len(signal)
        if L < 8:
            return signal.copy()
        try:
            wavelet = pywt.Wavelet(self.cfg.wavelet)
            max_level = pywt.dwt_max_level(L, wavelet.dec_len)
        except Exception:
            return signal.copy()
        level = min(self.cfg.wavelet_level, max_level)
        if level < 1:
            return signal.copy()
        try:
            coeffs = pywt.wavedec(signal, self.cfg.wavelet, level=level)
        except Exception:
            return signal.copy()

        finest = coeffs[-1]
        if len(finest) == 0:
            return signal.copy()
        sigma = float(np.median(np.abs(finest)) / 0.6745)
        if sigma < 1e-8:
            return signal.copy()
        thr = sigma * math.sqrt(2.0 * math.log(L))

        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, thr, mode="soft"))
        try:
            rec = pywt.waverec(new_coeffs, self.cfg.wavelet)
        except Exception:
            return signal.copy()
        return np.asarray(rec[:L], dtype=np.float32)

    def process(self, strengths, L):
        strength = np.asarray(strengths, dtype=np.float32)
        sig = self._normalize(self._resample(strength, L))
        imfs, residual = self._emd(sig)
        x_low, x_mid, x_high = self._group_bands(imfs, residual)
        x_high_dn = self._wavelet_denoise(x_high)
        return {"low": x_low, "mid": x_mid, "high": x_high_dn}


# =====================================================================
# 3. Dataset
# =====================================================================
class SeqDataset(Dataset):
    """
    Each user yields one sample:
        history = seq[:-1]   (truncated / padded to cfg.seq_len)
        target  = seq[-1]
    """

    def __init__(self, sequences, strengths, config, signal_processor,
                 user_subset=None, desc=""):
        self.samples = []
        self.cfg = config
        users = sequences.keys() if user_subset is None else user_subset
        users = [u for u in users if len(sequences[u]) >= 2]

        for u in tqdm(users, desc=desc or "preprocessing"):
            items = sequences[u]
            strg = strengths[u]
            if len(items) < 2:
                continue
            hist_items = items[:-1]
            hist_str = strg[:-1]
            target = items[-1]

            L = config.seq_len
            if len(hist_items) >= L:
                item_seq = hist_items[-L:]
                hist_str = hist_str[-L:]
            else:
                pad = L - len(hist_items)
                item_seq = [0] * pad + hist_items
                hist_str = [0.5] * pad + hist_str  # neutral for padding

            signals = signal_processor.process(hist_str, L)
            self.samples.append({
                "user_id": u,
                "item_seq": np.asarray(item_seq, dtype=np.int64),
                "sig_low": signals["low"],
                "sig_mid": signals["mid"],
                "sig_high": signals["high"],
                "target": int(target),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "user_id": torch.tensor(s["user_id"], dtype=torch.long),
            "item_seq": torch.from_numpy(s["item_seq"]),
            "sig_low": torch.from_numpy(s["sig_low"]),
            "sig_mid": torch.from_numpy(s["sig_mid"]),
            "sig_high": torch.from_numpy(s["sig_high"]),
            "target": torch.tensor(s["target"], dtype=torch.long),
        }


# =====================================================================
# 4. Graph construction
# =====================================================================
def build_norm_adj(sequences, n_users, n_items_total):
    """Symmetric-normalized user-item bipartite graph. n_items_total includes padding."""
    rows, cols, vals = [], [], []
    for u, items in sequences.items():
        for i in set(items):
            rows.append(u); cols.append(n_users + i); vals.append(1.0)
            rows.append(n_users + i); cols.append(u); vals.append(1.0)
    A = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_users + n_items_total, n_users + n_items_total),
        dtype=np.float32,
    )
    deg = np.asarray(A.sum(axis=1)).flatten()
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    A_norm = sp.diags(inv_sqrt) @ A @ sp.diags(inv_sqrt)
    return A_norm.tocoo()


# =====================================================================
# 5. FAME model
# =====================================================================
class FAME(nn.Module):
    def __init__(self, n_users, n_items_total, cfg, adj_norm):
        super().__init__()
        self.cfg = cfg
        self.n_users = n_users
        self.n_items = n_items_total
        d = cfg.embed_dim
        dh = cfg.hidden_dim

        self.item_embedding = nn.Embedding(n_items_total, d, padding_idx=0)
        self.user_embedding = nn.Embedding(n_users, d)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

        self.W_low = nn.Linear(1, d, bias=True)
        self.W_mid = nn.Linear(1, d, bias=True)
        self.W_high = nn.Linear(1, d, bias=True)

        lstm_kwargs = dict(
            input_size=2 * d,
            hidden_size=dh,
            num_layers=cfg.n_layers_lstm,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers_lstm > 1 else 0.0,
        )
        self.lstm_low = nn.LSTM(**lstm_kwargs)
        self.lstm_mid = nn.LSTM(**lstm_kwargs)
        self.lstm_high = nn.LSTM(**lstm_kwargs)

        self.register_buffer("adj_idx",
                             torch.tensor(np.vstack([adj_norm.row, adj_norm.col]),
                                          dtype=torch.long))
        self.register_buffer("adj_val",
                             torch.tensor(adj_norm.data, dtype=torch.float32))

        self.gate_W = nn.Linear(4 * dh, 4, bias=True)
        nn.init.zeros_(self.gate_W.bias)

    def _graph_propagate(self):
        X = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        A = torch.sparse_coo_tensor(
            self.adj_idx, self.adj_val,
            (self.n_users + self.n_items, self.n_users + self.n_items),
            device=X.device,
        ).coalesce()
        for _ in range(self.cfg.graph_layers):
            X = torch.sparse.mm(A, X)
        return X[: self.n_users], X[self.n_users:]

    def _run_branch(self, lstm, W, item_emb, sig):
        proj = W(sig.unsqueeze(-1))               # (B, L, d)
        x = torch.cat([item_emb, proj], dim=-1)   # (B, L, 2d)
        _, (h, _) = lstm(x)
        return h[-1]                              # (B, dh)

    def forward(self, user_ids, item_seq, sig_low, sig_mid, sig_high):
        item_emb = self.item_embedding(item_seq)

        h_low = self._run_branch(self.lstm_low, self.W_low, item_emb, sig_low)
        h_mid = self._run_branch(self.lstm_mid, self.W_mid, item_emb, sig_mid)
        h_high = self._run_branch(self.lstm_high, self.W_high, item_emb, sig_high)

        user_prop, item_prop = self._graph_propagate()
        h_g = user_prop[user_ids]

        cat = torch.cat([h_low, h_mid, h_high, h_g], dim=-1)
        g = torch.softmax(self.gate_W(cat), dim=-1)

        u = (g[:, 0:1] * h_low + g[:, 1:2] * h_mid
             + g[:, 2:3] * h_high + g[:, 3:4] * h_g)

        v_graph = self.item_embedding.weight + item_prop
        scores = u @ v_graph.t()

        return {
            "scores": scores,
            "h_low": h_low, "h_mid": h_mid, "h_high": h_high,
            "h_g": h_g, "u": u, "v_graph": v_graph, "gate": g,
        }

    def branch_diversity_loss(self, h_low, h_mid, h_high, h_g):
        hs = [h_low, h_mid, h_high, h_g]
        loss, count = 0.0, 0
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                cos = F.cosine_similarity(hs[i], hs[j], dim=-1)
                loss = loss + torch.clamp(cos - self.cfg.delta, min=0.0).mean()
                count += 1
        return loss / max(count, 1)


# =====================================================================
# 6. Sampling
# =====================================================================
def build_item_sampling_dist(sequences, n_items_total, pop_exp):
    counts = np.zeros(n_items_total, dtype=np.float64)
    for items in sequences.values():
        for i in items:
            if 0 < i < n_items_total:
                counts[i] += 1.0
    counts[0] = 0.0
    weights = counts ** pop_exp
    total = weights.sum()
    if total <= 0:
        weights[1:] = 1.0
        total = weights.sum()
    return torch.tensor(weights / total, dtype=torch.float32)


# =====================================================================
# 7. DPP-style greedy re-ranking
# =====================================================================
def dpp_rerank(scores, item_embeds, top_k=10, n_candidates=50, lam=0.3, sigma=1.0):
    scores = np.asarray(scores)
    item_embeds = np.asarray(item_embeds)

    cand = np.argsort(-scores)[:n_candidates]
    cand_scores = scores[cand]
    cand_emb = item_embeds[cand]

    diff = cand_emb[:, None, :] - cand_emb[None, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    K = np.exp(-dist2 / (2.0 * sigma ** 2))

    selected, chosen_idx = [], []
    remaining = list(range(len(cand)))

    for _ in range(min(top_k, len(cand))):
        best_j, best_val = None, -1e18
        for j in remaining:
            if not chosen_idx:
                cond_var = K[j, j]
            else:
                K_LL = K[np.ix_(chosen_idx, chosen_idx)]
                K_jL = K[j, chosen_idx]
                try:
                    sol = np.linalg.solve(K_LL, K_jL)
                    cond_var = K[j, j] - float(K_jL @ sol)
                except np.linalg.LinAlgError:
                    cond_var = K[j, j]
            cond_var = max(cond_var, 1e-8)
            val = cand_scores[j] + lam * math.log(cond_var)
            if val > best_val:
                best_val, best_j = val, j
        selected.append(int(cand[best_j]))
        chosen_idx.append(best_j)
        remaining.remove(best_j)

    return selected


# =====================================================================
# 8. Training
# =====================================================================
def train_epoch(model, loader, optimizer, item_dist, cfg):
    model.train()
    tot_loss = tot_rel = tot_div = 0.0
    n_batches = 0

    for batch in loader:
        user_ids = batch["user_id"].to(cfg.device)
        item_seq = batch["item_seq"].to(cfg.device)
        sig_low = batch["sig_low"].to(cfg.device)
        sig_mid = batch["sig_mid"].to(cfg.device)
        sig_high = batch["sig_high"].to(cfg.device)
        targets = batch["target"].to(cfg.device)

        out = model(user_ids, item_seq, sig_low, sig_mid, sig_high)
        scores = out["scores"]
        B = scores.size(0)

        pos_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
        neg_ids = torch.multinomial(item_dist, B * cfg.n_neg, replacement=True)\
                       .view(B, cfg.n_neg).to(cfg.device)
        neg_scores = scores.gather(1, neg_ids)

        rel_loss = (
            F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        )
        div_loss = model.branch_diversity_loss(
            out["h_low"], out["h_mid"], out["h_high"], out["h_g"]
        )
        loss = rel_loss + cfg.beta * div_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        tot_loss += loss.item(); tot_rel += rel_loss.item()
        tot_div += div_loss.item(); n_batches += 1

    n = max(n_batches, 1)
    return tot_loss / n, tot_rel / n, tot_div / n


# =====================================================================
# 9. Evaluation
# =====================================================================
@torch.no_grad()
def evaluate(model, loader, n_items_total, cfg, train_items_per_user,
             k_list=(5, 10)):
    model.eval()
    ndcg = {k: [] for k in k_list}
    recall = {k: [] for k in k_list}
    topk_collect = {k: [] for k in k_list}
    ild_vals, novelty_vals = [], []

    v = model.item_embedding.weight.detach().cpu().numpy()
    pop = np.zeros(n_items_total, dtype=np.float64)
    for items in train_items_per_user.values():
        for i in items:
            if 0 < i < n_items_total:
                pop[i] += 1.0
    pop_prob = (pop + 1e-8) / (pop.sum() + 1e-8 * n_items_total)

    for batch in loader:
        user_ids = batch["user_id"].to(cfg.device)
        item_seq = batch["item_seq"].to(cfg.device)
        sig_low = batch["sig_low"].to(cfg.device)
        sig_mid = batch["sig_mid"].to(cfg.device)
        sig_high = batch["sig_high"].to(cfg.device)
        targets = batch["target"].cpu().numpy()

        out = model(user_ids, item_seq, sig_low, sig_mid, sig_high)
        scores = out["scores"].cpu().numpy()
        scores[:, 0] = -1e9

        for b in range(scores.shape[0]):
            uid = int(user_ids[b].item())
            for i in train_items_per_user.get(uid, set()):
                if 0 < i < n_items_total:
                    scores[b, i] = -1e9

        for k in k_list:
            topk = np.argsort(-scores, axis=1)[:, :k]
            topk_collect[k].append(topk)
            for b in range(scores.shape[0]):
                t = targets[b]
                if t in topk[b]:
                    rank = int(np.where(topk[b] == t)[0][0])
                    ndcg[k].append(1.0 / math.log2(rank + 2))
                    recall[k].append(1.0)
                else:
                    ndcg[k].append(0.0); recall[k].append(0.0)

        top10 = np.argsort(-scores, axis=1)[:, :10]
        for b in range(top10.shape[0]):
            embs = v[top10[b]]
            dists = [np.linalg.norm(embs[i] - embs[j])
                     for i in range(10) for j in range(i + 1, 10)]
            ild_vals.append(np.mean(dists) if dists else 0.0)
            for i in top10[b]:
                novelty_vals.append(-math.log2(pop_prob[i] + 1e-12))

    res = {}
    for k in k_list:
        res[f"NDCG@{k}"] = float(np.mean(ndcg[k]))
        res[f"Recall@{k}"] = float(np.mean(recall[k]))
    res["ILD"] = float(np.mean(ild_vals))
    top10_all = np.concatenate(topk_collect[10], axis=0).flatten()
    res["Coverage@10"] = float(len(set(top10_all.tolist())) / n_items_total)
    res["Novelty"] = float(np.mean(novelty_vals))
    return res


@torch.no_grad()
def evaluate_with_dpp(model, loader, n_items_total, cfg, train_items_per_user,
                      lam=0.3, top_k=10, n_candidates=50):
    model.eval()
    ndcg, recall, ild_vals = [], [], []
    all_topk = []
    v = model.item_embedding.weight.detach().cpu().numpy()

    for batch in loader:
        user_ids = batch["user_id"].to(cfg.device)
        item_seq = batch["item_seq"].to(cfg.device)
        sig_low = batch["sig_low"].to(cfg.device)
        sig_mid = batch["sig_mid"].to(cfg.device)
        sig_high = batch["sig_high"].to(cfg.device)
        targets = batch["target"].cpu().numpy()

        out = model(user_ids, item_seq, sig_low, sig_mid, sig_high)
        scores = out["scores"].cpu().numpy()
        scores[:, 0] = -1e9
        for b in range(scores.shape[0]):
            uid = int(user_ids[b].item())
            for i in train_items_per_user.get(uid, set()):
                if 0 < i < n_items_total:
                    scores[b, i] = -1e9

        for b in range(scores.shape[0]):
            ranked = dpp_rerank(scores[b], v, top_k=top_k,
                                n_candidates=n_candidates, lam=lam)
            all_topk.append(ranked)
            t = targets[b]
            if t in ranked:
                r = ranked.index(t)
                ndcg.append(1.0 / math.log2(r + 2)); recall.append(1.0)
            else:
                ndcg.append(0.0); recall.append(0.0)
            embs = v[ranked]
            d = [np.linalg.norm(embs[i] - embs[j])
                 for i in range(len(ranked)) for j in range(i + 1, len(ranked))]
            ild_vals.append(np.mean(d) if d else 0.0)

    flat = [i for lst in all_topk for i in lst]
    return {
        "NDCG@10_DPP": float(np.mean(ndcg)),
        "Recall@10_DPP": float(np.mean(recall)),
        "ILD_DPP": float(np.mean(ild_vals)),
        "Coverage@10_DPP": float(len(set(flat)) / n_items_total),
        "lambda": lam,
    }


# =====================================================================
# 10. Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-1m", "beauty"], default="ml-1m")
    parser.add_argument("--data-root", type=str, default=r"D:/datasets",
                        help="Directory containing ratings.dat or Beauty_5.json.gz")
    parser.add_argument("--epochs", type=int, default=Config.n_epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--lambda-dpp", type=float, default=0.3)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    cfg.n_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.seq_len = args.seq_len

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"[FAME] device={cfg.device}  dataset={args.dataset}  "
          f"data_root={args.data_root}")

    # ---- Load real data ----
    sequences, strengths, n_users, n_items = prepare_dataset(
        args.dataset, args.data_root
    )
    n_items_total = n_items + 1  # item 0 = padding

    # ---- 80/10/10 user split ----
    all_users = sorted(sequences.keys())
    random.shuffle(all_users)
    n_tr = int(0.8 * len(all_users))
    n_va = int(0.1 * len(all_users))
    train_users = all_users[:n_tr]
    val_users = all_users[n_tr:n_tr + n_va]
    test_users = all_users[n_tr + n_va:]

    print(f"[FAME] split: train={len(train_users)}  "
          f"val={len(val_users)}  test={len(test_users)} users")

    sp_processor = SignalProcessor(cfg)
    train_ds = SeqDataset(sequences, strengths, cfg, sp_processor,
                          user_subset=train_users, desc="train EMD")
    val_ds = SeqDataset(sequences, strengths, cfg, sp_processor,
                        user_subset=val_users, desc="val EMD")
    test_ds = SeqDataset(sequences, strengths, cfg, sp_processor,
                         user_subset=test_users, desc="test EMD")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0)

    # ---- Graph ----
    print("[FAME] building user-item graph ...")
    adj_norm = build_norm_adj(sequences, n_users, n_items_total)
    print(f"[FAME] graph: {n_users} users + {n_items_total} items, "
          f"{adj_norm.nnz // 2} undirected edges")

    # ---- Model ----
    model = FAME(n_users, n_items_total, cfg, adj_norm).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    item_dist = build_item_sampling_dist(sequences, n_items_total, cfg.pop_exp)\
                    .to(cfg.device)
    train_items_per_user = {u: set(sequences[u][:-1]) for u in train_users}

    # ---- Train with early stopping ----
    best_val, best_state, wait = -1.0, None, 0
    for epoch in range(1, cfg.n_epochs + 1):
        tr_loss, tr_rel, tr_div = train_epoch(model, train_loader, optimizer,
                                              item_dist, cfg)
        val_metrics = evaluate(model, val_loader, n_items_total, cfg,
                               train_items_per_user)
        print(f"epoch {epoch:03d} | loss {tr_loss:.4f} "
              f"(rel {tr_rel:.4f}, div {tr_div:.4f}) | "
              f"val N@10 {val_metrics['NDCG@10']:.4f} "
              f"R@10 {val_metrics['Recall@10']:.4f} "
              f"ILD {val_metrics['ILD']:.4f} "
              f"Cov@10 {val_metrics['Coverage@10']:.4f}")

        if val_metrics["NDCG@10"] > best_val:
            best_val = val_metrics["NDCG@10"]
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"[FAME] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Test: relevance ranking (λ=0) ----
    test_metrics = evaluate(model, test_loader, n_items_total, cfg,
                            train_items_per_user)
    print("\n[Test | λ=0]")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # ---- Test: DPP re-ranking ----
    dpp_metrics = evaluate_with_dpp(
        model, test_loader, n_items_total, cfg, train_items_per_user,
        lam=args.lambda_dpp, top_k=10, n_candidates=50,
    )
    print(f"\n[Test | DPP λ={args.lambda_dpp}]")
    for k, v in dpp_metrics.items():
        print(f"  {k:16s}: {v:.4f}")

    # ---- Gating distribution ----
    model.eval()
    gates = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                batch["user_id"].to(cfg.device),
                batch["item_seq"].to(cfg.device),
                batch["sig_low"].to(cfg.device),
                batch["sig_mid"].to(cfg.device),
                batch["sig_high"].to(cfg.device),
            )
            gates.append(out["gate"].cpu().numpy())
    if gates:
        gates = np.concatenate(gates, axis=0)
        print("\n[Average gating weights]")
        for name, w in zip(["low", "mid", "high", "graph"], gates.mean(axis=0)):
            print(f"  {name:6s}: {w:.4f}")


if __name__ == "__main__":
    main()