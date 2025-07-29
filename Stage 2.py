import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

# TPU support
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(" Using TPU:", device)
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" TPU unavailable, using:", device)

# -----------------------------
# Parameters
# -----------------------------
INPUT_PATH   = ""
OUTPUT_PATH  = ""
MAX_ROWS     =       
BATCH_SIZE   = 
INPUT_DIM    = None              # inferred below
CALIB_DIM    = 64                # final embedding size
HIDDEN_DIM   = 256

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# -----------------------------
# Model Components
# -----------------------------
class ExpertFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class NaturalCalibrator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.fc(x))

class FractalCalibrator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.block(x)

class MoCaERouter(nn.Module):
    def __init__(self, expert_ffn, temperature=0.7):
        super().__init__()
        self.expert = expert_ffn
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.temperature = temperature

    def forward(self, x):
        w = F.softmax(self.gamma / self.temperature, dim=0)[0]
        return self.expert(x) * w

# -----------------------------
# Load & Prepare Data
# -----------------------------
npz = np.load(INPUT_PATH, mmap_mode="r")
key = list(npz.keys())[0]
arr = npz[key]
print("Original shape:", arr.shape)

# If flattened 1D, reshape to (-1, 1)
if arr.ndim == 1:
    arr = arr.reshape(-1, 1)

N_full, INPUT_DIM = arr.shape
N = min(N_full, MAX_ROWS)
arr = arr[:N]
print(f"Processing {N:,} rows × {INPUT_DIM} dims")

# -----------------------------
# Instantiate Modules
# -----------------------------
expert_ffn  = ExpertFFN(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=CALIB_DIM).to(device)
router      = MoCaERouter(expert_ffn).to(device)
nat_cal     = NaturalCalibrator(CALIB_DIM).to(device)
fract_cal   = FractalCalibrator(CALIB_DIM).to(device)

# -----------------------------
# Allocate HDF5 Dataset
# -----------------------------
with h5py.File(OUTPUT_PATH, "w") as hf:
    dset = hf.create_dataset(
        "calibrated_embeddings",
        shape=(N, CALIB_DIM),
        dtype="float16",
        chunks=(BATCH_SIZE, CALIB_DIM),
        compression="gzip",
        compression_opts=4,
    )

    # -----------------------------
    # Batch Processing Loop
    # -----------------------------
    offset = 0
    with torch.no_grad():
        while offset < N:
            end = min(N, offset + BATCH_SIZE)
            chunk = arr[offset:end]                    # memmap slice
            x = torch.from_numpy(chunk).float().to(device)

            # forward through router + calibrators
            fused      = router(x)
            nat_adj    = nat_cal(fused)
            fract_adj  = fract_cal(fused)
            calibrated = (fused + nat_adj + fract_adj).half()

            if 'xm' in globals():
                xm.mark_step()  # sync TPU

            # write back to HDF5 (on CPU)
            dset[offset:end, :] = calibrated.cpu().numpy()
            print(f"Processed rows {offset:,}–{end:,}")
            offset = end

print("All done — embeddings saved to", OUTPUT_PATH)
