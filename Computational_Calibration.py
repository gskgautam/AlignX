import os
import time
import psutil
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
# Paths for TruthfulQA
# ─────────────────────────────────────────────────────────────────────────────
EMB_H5        = ""
TRAIN_CSV    = ""
TEST_CSV     = ""

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load & slice embeddings
# ─────────────────────────────────────────────────────────────────────────────
with h5py.File(EMB_H5, 'r') as hf:
    full_emb = hf['calibrated_embeddings'][:]   # (N_full × D)
print(f"Loaded embeddings: {full_emb.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load labels from CSVs
# ─────────────────────────────────────────────────────────────────────────────
df_tr = pd.read_csv(TRAIN_CSV)
df_te = pd.read_csv(TEST_CSV)
# ensure there is a 'label' column (0/1 or class index)
if 'label' not in df_tr.columns:
    raise KeyError(f"'label' column not found in {TRAIN_CSV}")
if 'label' not in df_te.columns:
    raise KeyError(f"'label' column not found in {TEST_CSV}")
y_tr = df_tr['label'].to_numpy(dtype=int)
y_te = df_te['label'].to_numpy(dtype=int)
n_tr, n_te = len(y_tr), len(y_te)
print(f"Train size: {n_tr}, Test size: {n_te} → total {n_tr + n_te}")

# Slice the first (n_tr + n_te) embeddings
emb = full_emb[: n_tr + n_te]
emb_tr, emb_te = emb[:n_tr], emb[n_tr:]
print(f"Sliced embeddings: train={emb_tr.shape}, test={emb_te.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3) Scoring functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc  = (pred == labels).astype(float)
    ece = 0.
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if mask.any():
            ece += abs(conf[mask].mean() - acc[mask].mean()) * mask.sum() / len(labels)
    return ece

def compute_brier(probs, labels):
    N,C = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), labels] = 1
    return np.mean(((probs - onehot)**2).sum(axis=1))

def temperature_scale(probs, T=1.0):
    logits = np.log(np.clip(probs,1e-12,1.0)) / T
    exp    = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Zero‑Shot Evaluation
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Xte = torch.tensor(emb_te, dtype=torch.float32).to(device)

t0 = time.time()
# Here we treat the 64‑dim embedding as logits directly
# If True labels are binary 0/1, slice to 2‑class softmax for zero‑shot:
if len(np.unique(y_tr)) == 2:
    # Map embedding to 2‑class via linear layer trained on train set
    # For a genuine zero‑shot without training, you need a pretrained model.
    # Here we simulate by training logistic regression on train set then eval on test
    # (this becomes few‑shot).
    pass

# Instead, for pure zero‑shot we compute softmax over embedding dims (not meaningful here)
logits_zs = Xte
probs_zs  = torch.softmax(logits_zs, dim=1).cpu().numpy()
infer_time_zs = time.time() - t0

ece_zs   = compute_ece(probs_zs, y_te)
ece_t_zs = compute_ece(temperature_scale(probs_zs), y_te)
brier_zs = compute_brier(probs_zs, y_te)

print("\n--- ZERO‑SHOT (softmax over 64‑dim embed) ---")
print(f"ECE               : {ece_zs:.4f}")
print(f"ECE‑t             : {ece_t_zs:.4f}")
print(f"Brier             : {brier_zs:.4f}")
print(f"Inference_Time_s  : {infer_time_zs:.4f}")
print(f"Train_Time_s      : {0.0:.4f}")
print(f"Train_Memory_MB   : {0.0:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5) Few‑Shot Evaluation
# ─────────────────────────────────────────────────────────────────────────────
Xtr = torch.tensor(emb_tr, dtype=torch.float32).to(device)
ytr = torch.tensor(y_tr, dtype=torch.long).to(device)

model = nn.Linear(Xtr.shape[1], len(np.unique(y_tr))).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

t1 = time.time()
model.train()
for _ in range(5):
    optimizer.zero_grad()
    logits = model(Xtr)
    loss   = criterion(logits, ytr)
    loss.backward()
    optimizer.step()
train_time = time.time() - t1
train_mem  = psutil.Process(os.getpid()).memory_info().rss / 1024**2

t2 = time.time()
model.eval()
with torch.no_grad():
    probs_fs = torch.softmax(model(Xte), dim=1).cpu().numpy()
infer_time_fs = time.time() - t2

ece_fs   = compute_ece(probs_fs, y_te)
ece_t_fs = compute_ece(temperature_scale(probs_fs), y_te)
brier_fs = compute_brier(probs_fs, y_te)

print("\n--- FEW‑SHOT (train→test) ---")
print(f"ECE               : {ece_fs:.4f}")
print(f"ECE‑t             : {ece_t_fs:.4f}")
print(f"Brier             : {brier_fs:.4f}")
print(f"Inference_Time_s  : {infer_time_fs:.4f}")
print(f"Train_Time_s      : {train_time:.4f}")
print(f"Train_Memory_MB   : {train_mem:.4f}")
