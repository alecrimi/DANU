"""
DANU on Cork neonatal HIE EEG — full DANU plus baseline comparison.

Uses overlapping windows with a 256-sample stride (1.28 s at 200 Hz; 50% overlap).
Models: full DANU, Tractable, GRU, and LSTM. The two DANU ablation variants
(noGate and noWTA) are excluded.
"""

import os
import re
import warnings
from math import gcd

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, signal

warnings.filterwarnings("ignore")

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = "EDF_format"
METADATA_FILE = "metadata.csv"
TARGET_FS = 200
GRADES = [1, 2, 3, 4]
SEQ_LEN = 512
WINDOW_STRIDE = 256  # 1.28 s at 200 Hz; 50% overlap
HORIZON = 300
HIDDEN_SIZE = 128
NUM_SEEDS = 5
MAX_EPOCHS = 400
PATIENCE = 45
BATCH_SIZE = 128
MAX_WINDOWS = 4000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR =  "cork_danu_stride256_results"
os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4] #, 5, 6, 7, 8, 9]

# Model names
FULL = "DANU"
BASELINES = ["Tractable", "GRU", "LSTM"]
MODELS = [FULL] + BASELINES

# EEG preprocessing settings
LOW_CUT_HZ = 0.5
HIGH_CUT_HZ = 40.0
NOTCH_HZ = None             # band-pass ends at 40 Hz, so 50 Hz notch is redundant
RECORDING_MINUTES = 10
EPS = 1e-8

BIPOLAR_PAIRS = [
    ("F4", "C4"),
    ("C4", "O2"),
    ("F3", "C3"),
    ("C3", "O1"),
    ("T4", "C4"),
    ("C4", "Cz"),
    ("Cz", "C3"),
    ("C3", "T3"),
]
EXPECTED_ELECTRODES = sorted({ch for pair in BIPOLAR_PAIRS for ch in pair})


def normalize_channel_name(name):
    s = str(name).strip().upper()
    s = re.sub(r"\b(EEG|REF|LE|RE|AVG|AVERAGE|CAR)\b", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    aliases = {
        "T7": "T3",
        "T3": "T3",
        "T8": "T4",
        "T4": "T4",
        "CZ": "CZ",
        "FCZ": "FCZ",
    }
    return aliases.get(s, s)


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# EDF LOADING + PREPROCESSING
# ─────────────────────────────────────────────
def find_electrode_indices(ch_names):
    normalized = [normalize_channel_name(n) for n in ch_names]
    mapping = {}
    missing = []
    for electrode in EXPECTED_ELECTRODES:
        matches = [i for i, name in enumerate(normalized)
                   if name == electrode.upper()]
        if len(matches) == 1:
            mapping[electrode] = matches[0]
        elif len(matches) == 0:
            missing.append(electrode)
        else:
            raise ValueError(
                f"Ambiguous EDF channels for {electrode}: "
                f"{[ch_names[i] for i in matches]}"
            )
    if missing:
        raise ValueError(
            f"Missing required electrodes {missing}. "
            f"EDF channels are: {list(ch_names)}. "
            "Check channel labels/aliases before proceeding."
        )
    return mapping


def apply_filter(data, fs):
    if data.ndim != 2:
        raise ValueError(f"Expected [channels, samples], got {data.shape}")
    nyquist = fs / 2.0
    high = min(HIGH_CUT_HZ, nyquist * 0.95)
    low = LOW_CUT_HZ
    if low >= high:
        raise ValueError(f"Invalid filter band {low}-{high} Hz for fs={fs}")
    sos = signal.butter(N=4, Wn=[low, high], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=1)
    if NOTCH_HZ is not None and NOTCH_HZ < nyquist:
        b, a = signal.iirnotch(w0=NOTCH_HZ, Q=30.0, fs=fs)
        filtered = signal.filtfilt(b, a, filtered, axis=1)
    return filtered


def resample_signal(data, orig_fs, target_fs):
    orig_fs_i = int(round(float(orig_fs)))
    target_fs_i = int(round(float(target_fs)))
    if abs(orig_fs_i - orig_fs) > 1e-3:
        n_out = int(round(data.shape[1] * target_fs / orig_fs))
        return signal.resample(data, n_out, axis=1)
    g = gcd(target_fs_i, orig_fs_i)
    return signal.resample_poly(data, target_fs_i // g, orig_fs_i // g, axis=1)


def load_and_preprocess_edf(path, target_fs=TARGET_FS):
    if not MNE_AVAILABLE:
        raise ImportError("MNE is required. Install with: pip install mne")

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    ch_names = raw.ch_names
    original_fs = float(raw.info["sfreq"])

    data = raw.get_data()
    idx = find_electrode_indices(ch_names)
    electrode_data = {
        electrode: data[channel_idx].astype(np.float64, copy=True)
        for electrode, channel_idx in idx.items()
    }

    bipolar = np.stack([
        electrode_data[a] - electrode_data[b]
        for a, b in BIPOLAR_PAIRS
    ], axis=0) * 1e6

    bipolar = apply_filter(bipolar, original_fs)
    if abs(original_fs - target_fs) > 1e-6:
        bipolar = resample_signal(bipolar, original_fs, target_fs)

    n_keep = min(bipolar.shape[1], int(round(target_fs * 60 * RECORDING_MINUTES)))
    bipolar = bipolar[:, :n_keep]

    if not np.isfinite(bipolar).all():
        raise ValueError("Non-finite values found after filtering/resampling")

    channel_std = bipolar.std(axis=1)
    if np.any(channel_std < 1e-10):
        bad = [BIPOLAR_PAIRS[i] for i in np.where(channel_std < 1e-10)[0]]
        raise ValueError(f"Flat/near-flat bipolar channels: {bad}")

    mean = bipolar.mean(axis=1, keepdims=True)
    std = bipolar.std(axis=1, keepdims=True)
    bipolar = (bipolar - mean) / np.maximum(std, EPS)

    info = {
        "path": path,
        "channel_names": ch_names,
        "original_fs": original_fs,
        "target_fs": target_fs,
        "n_samples": int(bipolar.shape[1]),
        "n_channels": int(bipolar.shape[0]),
        "bipolar_pairs": [f"{a}-{b}" for a, b in BIPOLAR_PAIRS],
    }
    return bipolar.astype(np.float32), info


def load_subjects(data_dir, metadata_file, target_fs=TARGET_FS, grades=None):
    meta = pd.read_csv(metadata_file)
    meta.columns = [c.strip().lower() for c in meta.columns]
    for col in ("sampling_freq", "grade", "file_id"):
        if col not in meta.columns:
            raise ValueError(f"Expected {col} column; got {list(meta.columns)}")

    meta_use = meta.copy()
    if grades is not None:
        meta_use = meta_use[meta_use["grade"].isin(grades)]
    print(f"[Data] {len(meta_use)} metadata rows selected for grades {grades}")

    grade_data = {g: [] for g in (grades or sorted(meta_use["grade"].unique()))}
    loaded, skipped = 0, 0
    qc_rows = []

    for _, row in meta_use.iterrows():
        file_id = str(row["file_id"]).strip()
        grade = int(row["grade"])
        candidates = [
            os.path.join(data_dir, f"{file_id}.edf"),
            os.path.join(data_dir, f"{file_id}.EDF"),
            os.path.join(data_dir, f"{file_id}_epoch1.edf"),
            os.path.join(data_dir, f"{file_id}_epoch1.EDF"),
        ]
        edf_path = next((p for p in candidates if os.path.exists(p)), None)
        if edf_path is None:
            print(f"  [WARN] EDF not found for {file_id} — skipping")
            skipped += 1
            continue
        try:
            processed, info = load_and_preprocess_edf(edf_path, target_fs)
            grade_data.setdefault(grade, []).append(processed)
            qc_rows.append({
                "file_id": file_id,
                "grade": grade,
                "path": edf_path,
                "original_fs": info["original_fs"],
                "target_fs": info["target_fs"],
                "n_channels": info["n_channels"],
                "n_samples": info["n_samples"],
                "bipolar_pairs": ";".join(info["bipolar_pairs"]),
            })
            loaded += 1
        except Exception as e:
            print(f"  [WARN] Error processing {edf_path}: {e}")
            skipped += 1

    pd.DataFrame(qc_rows).to_csv(
        os.path.join(SAVE_DIR, "preprocessing_qc.csv"), index=False
    )
    print(f"[Data] Loaded {loaded} subjects, skipped {skipped}")
    for g, recordings in grade_data.items():
        print(f"  Grade {g}: {len(recordings)} subjects")
    return grade_data


# ─────────────────────────────────────────────
# DATASET — multi-step-ahead prediction
# ─────────────────────────────────────────────
class EEGWindowDataset(Dataset):
    """x: [SEQ_LEN, n_ch]; y: [n_ch] at t + HORIZON steps."""
    def __init__(self, recordings, seq_len=SEQ_LEN, horizon=HORIZON,
                 max_windows=None, rng=None):
        xs, ys = [], []
        stride = WINDOW_STRIDE
        for rec in recordings:
            n_ch, T = rec.shape
            last_start = T - seq_len - horizon
            for start in range(0, last_start + 1, stride):
                xs.append(rec[:, start:start + seq_len].T.astype(np.float32))
                ys.append(rec[:, start + seq_len + horizon - 1].astype(np.float32))

        if not xs:
            raise ValueError(
                "No EEG windows were generated. Check recording duration, "
                "SEQ_LEN, HORIZON, and subject counts."
            )
        xs = np.stack(xs)
        ys = np.stack(ys)
        if max_windows is not None and len(xs) > max_windows:
            rng = rng or np.random.default_rng(42)
            idx = rng.choice(len(xs), max_windows, replace=False)
            xs, ys = xs[idx], ys[idx]

        self.x = torch.from_numpy(xs)
        self.y = torch.from_numpy(ys)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def make_loaders(recordings, seed, val_ratio=0.2, horizon=HORIZON):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recordings))
    n_val = max(1, int(len(recordings) * val_ratio))
    val_recs = [recordings[i] for i in idx[:n_val]]
    train_recs = [recordings[i] for i in idx[n_val:]]

    train_ds = EEGWindowDataset(
        train_recs, horizon=horizon, max_windows=MAX_WINDOWS,
        rng=np.random.default_rng(seed)
    )
    val_ds = EEGWindowDataset(val_recs, horizon=horizon)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
class TractableDendriticCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments=4):
        super().__init__()
        self.K = num_compartments
        self.comp_size = hidden_size // num_compartments
        assert hidden_size % num_compartments == 0
        self.W_in = nn.ModuleList([
            nn.Linear(input_size, self.comp_size) for _ in range(self.K)
        ])
        self.W_rec = nn.ModuleList([
            nn.Linear(hidden_size, self.comp_size) for _ in range(self.K)
        ])
        self.integration = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        parts = [
            torch.tanh(self.W_in[i](x) + self.W_rec[i](h))
            for i in range(self.K)
        ]
        return torch.tanh(self.integration(torch.cat(parts, dim=1)))


class TractableDendriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_compartments=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TractableDendriticCell(input_size, hidden_size, num_compartments)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, truncate_k=None):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
            if truncate_k is not None and (t + 1) % truncate_k == 0:
                h = h.detach()
        return self.fc(h)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class DANUCell(nn.Module):
    """
    use_gate=False -> d_gated = d_tilde (no W_g, no d_bar)
    use_wta=False  -> uniform 1/K weights over compartments (no W_q, no softmax)
    With both flags True this is exactly the original DANU cell.
    """
    def __init__(self, input_size, hidden_size, num_compartments=2,
                 use_gate=True, use_wta=True):
        super().__init__()
        self.K = num_compartments
        self.H = hidden_size
        self.use_gate = use_gate
        self.use_wta = use_wta
        in_dim = input_size + hidden_size
        self.W_E = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_I = nn.Linear(in_dim, hidden_size * num_compartments)
        if use_gate:
            self.W_g = nn.Linear(in_dim + hidden_size, hidden_size * num_compartments)
        self.lambda_I = nn.Parameter(torch.ones(num_compartments, hidden_size) * 0.1)
        self.ln = nn.LayerNorm(hidden_size)
        self.beta = nn.Parameter(torch.ones(num_compartments, 1))
        self.theta = nn.Parameter(torch.zeros(num_compartments, 1))
        self.gamma = nn.Parameter(torch.ones(num_compartments, 1) * 0.1)
        if use_wta:
            self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        B = x.size(0)
        xh = torch.cat([x, h], dim=1)
        E = self.W_E(xh).view(B, self.K, self.H)
        I = self.W_I(xh).view(B, self.K, self.H)
        u = self.ln(E - self.lambda_I * I)
        sig = torch.sigmoid(self.beta * (u - self.theta))
        nmda = self.gamma * (u ** 2) / (1.0 + u ** 2)
        d_tilde = sig + nmda

        if self.use_gate:
            d_bar = d_tilde.mean(dim=1, keepdim=True)
            xh_dbar = torch.cat([xh, d_bar.squeeze(1)], dim=1)
            G = torch.sigmoid(self.W_g(xh_dbar).view(B, self.K, self.H))
            d_gated = G * d_tilde
        else:
            d_gated = d_tilde

        if self.use_wta:
            q = self.W_q(h).unsqueeze(2)
            scores = torch.bmm(d_gated, q).squeeze(2) / (self.H ** 0.5)
            alpha = torch.softmax(scores, dim=1)
            attended = (alpha.unsqueeze(2) * d_gated).sum(dim=1)
        else:
            attended = d_gated.mean(dim=1)

        return torch.tanh(self.ln_out(self.W_out(attended) + h))


class DANU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_compartments=5,
                 use_gate=True, use_wta=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DANUCell(input_size, hidden_size, num_compartments,
                             use_gate=use_gate, use_wta=use_wta)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, truncate_k=None):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
            if truncate_k is not None and (t + 1) % truncate_k == 0:
                h = h.detach()
        return self.fc(h)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# TRAINING (unchanged)
# ─────────────────────────────────────────────
def train_model(model, train_loader, val_loader, max_epochs=MAX_EPOCHS,
                patience=PATIENCE, lr=1e-4, clip=1.0, truncate_k=None,
                log_grad_norm=False):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs, eta_min=lr / 50
    )
    loss_fn = nn.MSELoss()
    best_val, patience_ctr, best_state = float("inf"), 0, None
    hist = {"stopped": max_epochs, "best_val": float("inf"),
            "train_loss": [], "val_loss": []}
    if log_grad_norm:
        hist["grad_norm"] = []

    def _forward(x):
        import inspect
        if "truncate_k" in inspect.signature(model.forward).parameters:
            return model(x, truncate_k=truncate_k)
        return model(x)

    for epoch in range(max_epochs):
        model.train()
        epoch_train_loss, n_train_batches = 0.0, 0
        epoch_grad_norm, n_grad_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(_forward(x), y)
            if torch.isfinite(loss):
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                epoch_train_loss += loss.item()
                n_train_batches += 1
                if log_grad_norm:
                    epoch_grad_norm += float(gn)
                    n_grad_batches += 1
        sched.step()
        train_loss = epoch_train_loss / n_train_batches if n_train_batches else float("nan")
        hist["train_loss"].append(train_loss)
        if log_grad_norm:
            hist["grad_norm"].append(
                epoch_grad_norm / n_grad_batches if n_grad_batches else float("nan")
            )

        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = loss_fn(_forward(x), y)
                if torch.isfinite(loss):
                    val_loss_sum += loss.item()
                    n_val_batches += 1
        val_loss = val_loss_sum / n_val_batches if n_val_batches else float("nan")
        hist["val_loss"].append(val_loss)

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                hist["stopped"] = epoch + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val"] = best_val
    return hist


# ─────────────────────────────────────────────
# PER-GRADE EXPERIMENT
# ─────────────────────────────────────────────
def build_models(n_ch):
    """Full DANU and three baselines; no DANU ablations."""
    return [
        (FULL, DANU(n_ch, HIDDEN_SIZE, n_ch, 5, use_gate=True, use_wta=True), 5e-5, 1.0),
        ("Tractable", TractableDendriticRNN(n_ch, HIDDEN_SIZE, n_ch, 4), 5e-5, 1.0),
        ("GRU", GRUModel(n_ch, HIDDEN_SIZE, n_ch), 5e-5, 1.0),
        ("LSTM", LSTMModel(n_ch, HIDDEN_SIZE, n_ch), 5e-5, 1.0),
    ]


def run_grade_experiment(grade, recordings, seeds=SEEDS, history_store=None):
    n_ch = recordings[0].shape[0]
    print(f"\n{'=' * 70}")
    print(f"  Grade {grade} | {len(recordings)} subjects | {n_ch} channels | {len(seeds)} seeds")
    print(f"{'=' * 70}")
    if len(recordings) < 3:
        print(f"  [SKIP] Not enough subjects (need >= 3, got {len(recordings)})")
        return None

    all_metrics = []
    for i, seed in enumerate(seeds):
        set_all_seeds(seed)
        train_loader, val_loader = make_loaders(recordings, seed)
        print(f"\n  Seed {i+1}/{len(seeds)} (value={seed}) "
              f"| train={len(train_loader.dataset)} | val={len(val_loader.dataset)}")

        seed_result = {}
        for name, model, lr, clip in build_models(n_ch):
            # Re-seed before each model so initialisation / shuffling of every
            # model depends only on the seed, not on which models ran before it.
            set_all_seeds(seed)
            if i == 0:
                print(f"    {name}: {n_params(model):,} params")
            print(f"    Training {name}...", end="", flush=True)
            h = train_model(model, train_loader, val_loader, lr=lr, clip=clip)
            seed_result[name] = h["best_val"]
            if history_store is not None:
                history_store.setdefault(name, []).append(h)
            print(f"  MSE={h['best_val']:.6f} (ep {h['stopped']})")
        all_metrics.append(seed_result)
    return all_metrics


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
def summarise(all_metrics):
    vals = {n: [m[n] for m in all_metrics] for n in MODELS}
    summary = {
        n: {"mean": float(np.mean(vals[n])), "std": float(np.std(vals[n], ddof=1))}
        for n in MODELS
    }
    # Paired comparisons: full DANU vs each baseline (positive improvement = DANU better).
    for other in BASELINES:
        _, p = stats.ttest_rel(vals[FULL], vals[other])
        improvement = np.mean([
            (bv - dv) / bv * 100 if bv != 0 else np.nan
            for dv, bv in zip(vals[FULL], vals[other])
        ])
        summary[f"p_danu_vs_{other}"] = float(p)
        summary[f"imp_vs_{other}"] = float(improvement)
    return summary, vals


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


def print_summary_table(grade_results):
    grades = sorted(grade_results.keys())
    print("\n" + "=" * 90)
    print("  RESULTS — Cork Neonatal EEG — Prediction MSE (same seeds/splits for all models)")
    print("=" * 90)
    for g in grades:
        s = grade_results[g]["summary"]
        print(f"\n  -- Grade {g} --")
        for model in MODELS:
            print(f"    {model:18s} MSE = {s[model]['mean']:.6f} ± {s[model]['std']:.6f}")
        print("    Full DANU vs baselines (+ = DANU better):")
        for b in BASELINES:
            p = s[f'p_danu_vs_{b}']
            print(f"      vs {b:10s} {s[f'imp_vs_{b}']:+.2f}%  p = {p:.4f} {sig_stars(p)}")


COLORS = {
    "DANU": "#1B4332", "Tractable": "#F18F01", "GRU": "#06A77D", "LSTM": "#A23B72",
}


def plot_grade_comparison(grade_results, save_path=None):
    grades = sorted(grade_results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        "DANU vs baseline models — Cork Neonatal EEG\n"
        "Prediction MSE by pathology grade (200 Hz), identical seeds and splits",
        fontsize=13, fontweight="bold"
    )
    x = np.arange(len(grades))
    width = 0.18
    offsets = np.linspace(-(len(MODELS)-1)/2, (len(MODELS)-1)/2, len(MODELS)) * width
    for i, model in enumerate(MODELS):
        means = [grade_results[g]["summary"][model]["mean"] for g in grades]
        stds = [grade_results[g]["summary"][model]["std"] for g in grades]
        ax.bar(x + offsets[i], means, width, yerr=stds, capsize=3,
               color=COLORS[model], alpha=0.9, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Grade {g}" for g in grades], fontsize=11)
    ax.set_ylabel("Prediction MSE (lower = better)")
    ax.set_title("Prediction error by grade", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot] Saved -> {save_path}")
    plt.close()


def plot_training_curves(grade_histories, save_path=None):
    grades = sorted(grade_histories.keys())
    n = len(grades)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.suptitle("Training curves (seed 0): solid = val, dashed = train",
                 fontsize=13, fontweight="bold")
    for gi, grade in enumerate(grades):
        ax = axes[0, gi]
        for name, hist_list in grade_histories[grade].items():
            if not hist_list:
                continue
            h = hist_list[0]
            epochs = np.arange(1, len(h["val_loss"]) + 1)
            ax.plot(epochs, h["val_loss"], color=COLORS.get(name), linewidth=2, label=f"{name} (val)")
            ax.plot(epochs, h["train_loss"], color=COLORS.get(name), linewidth=1,
                    linestyle="--", alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (log scale)")
        ax.set_title(f"Grade {grade}", fontweight="bold")
        ax.grid(alpha=0.3)
        if gi == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Plot] Saved -> {save_path}")
    plt.close()


def save_csv(grade_results):
    rows = []
    for g, r in grade_results.items():
        for model in MODELS:
            for seed, v in zip(SEEDS, r["vals"][model]):
                rows.append({"grade": g, "model": model, "seed": seed, "val_mse": v})
    pd.DataFrame(rows).to_csv(os.path.join(SAVE_DIR, "per_seed_mse.csv"), index=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DANU Cork EEG — full DANU vs baselines")
    print(f"  fs={TARGET_FS} Hz | bandpass={LOW_CUT_HZ}-{HIGH_CUT_HZ} Hz | "
          f"SEQ_LEN={SEQ_LEN} | HORIZON={HORIZON} | seeds={len(SEEDS)}")
    print(f"  models: {MODELS}")
    print("=" * 70)

    grade_data = load_subjects(DATA_DIR, METADATA_FILE, target_fs=TARGET_FS, grades=GRADES)
    grade_results = {}
    grade_histories = {}

    for grade in GRADES:
        recordings = grade_data.get(grade, [])
        if len(recordings) < 3:
            print(f"[WARN] Grade {grade}: insufficient usable recordings ({len(recordings)}); skipping")
            continue
        hist_store = {}
        metrics = run_grade_experiment(grade, recordings, seeds=SEEDS, history_store=hist_store)
        if metrics is None:
            continue
        summary, vals = summarise(metrics)
        grade_results[grade] = {"metrics": metrics, "summary": summary, "vals": vals}
        grade_histories[grade] = hist_store

    if not grade_results:
        print("[ERROR] No grades had sufficient usable data.")
        return

    print_summary_table(grade_results)
    save_csv(grade_results)
    plot_grade_comparison(grade_results, os.path.join(SAVE_DIR, "cork_danu_comparison.png"))
    plot_training_curves(grade_histories, os.path.join(SAVE_DIR, "cork_danu_training_curves.png"))
    print(f"\n[Done] Results saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
