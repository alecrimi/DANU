"""DANU Cork EEG ablations with CSV-driven experiment skipping.

intermediate_results.csv is the source of truth: an existing grade/seed/model
row is always skipped, regardless of whether a training curve exists.
New runs save their curves atomically after every epoch. Checkpoints are
intentionally ignored; old CSV results are never retrained to recreate curves.
"""

import os
import re
import json
import random
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


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = "EDF_format"
METADATA_FILE = "metadata.csv"
TARGET_FS = 200
GRADES = [1, 2, 3, 4]
SEQ_LEN = 512
HORIZON = 300
HIDDEN_SIZE = 128
MAX_EPOCHS = 400
PATIENCE = 45
BATCH_SIZE = 128
MAX_WINDOWS = 4000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "cork_danu_ablation_results"
HISTORY_DIR = os.path.join(SAVE_DIR, "training_curves")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

FULL = "DANU"
ABLATIONS = ["DANU-noGate", "DANU-noWTA"]
BASELINES = ["Tractable", "GRU", "LSTM"]
MODELS = [FULL] + ABLATIONS + BASELINES

LOW_CUT_HZ = 0.5
HIGH_CUT_HZ = 40.0
NOTCH_HZ = None
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

# ---------------------------------------------------------------------
# REPRODUCIBILITY + CSV / CURVE PERSISTENCE
# ---------------------------------------------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def results_path():
    return os.path.join(SAVE_DIR, "intermediate_results.csv")


def load_intermediate_results():
    path = results_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=["grade", "seed", "model", "val_mse"])
    df = pd.read_csv(path)
    required = {"grade", "seed", "model", "val_mse"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    df = df.copy()
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["val_mse"] = pd.to_numeric(df["val_mse"], errors="coerce")
    df["model"] = df["model"].astype(str)
    df = df.dropna(subset=["grade", "seed", "val_mse"])
    # If legacy CSV contains duplicates, retain the last recorded row.
    return df.drop_duplicates(["grade", "seed", "model"], keep="last")


def result_exists(results, grade, seed, model_name):
    mask = ((results["grade"] == int(grade)) &
            (results["seed"] == int(seed)) &
            (results["model"] == str(model_name)))
    return bool(mask.any())


def save_intermediate_result(grade, seed, model_name, val_mse):
    """Atomically upsert one completed experiment result."""
    df = load_intermediate_results()
    row = {"grade": int(grade), "seed": int(seed),
           "model": str(model_name), "val_mse": float(val_mse)}
    mask = ((df["grade"] == row["grade"]) &
            (df["seed"] == row["seed"]) &
            (df["model"] == row["model"]))
    df = pd.concat([df.loc[~mask], pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["grade", "seed", "model"])
    tmp = results_path() + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, results_path())


def curve_path(grade, seed, model_name):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return os.path.join(HISTORY_DIR, f"grade{grade}_seed{seed}_{safe_name}.json")


def save_curve_history(grade, seed, model_name, history):
    path = curve_path(grade, seed, model_name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f)
    os.replace(tmp, path)


def load_seed0_curves(grade):
    histories = {}
    for model_name in MODELS:
        path = curve_path(grade, SEEDS[0], model_name)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    histories[model_name] = [json.load(f)]
            except (OSError, ValueError):
                pass
    return histories


# ---------------------------------------------------------------------
# EDF LOADING + PREPROCESSING
# ---------------------------------------------------------------------
def normalize_channel_name(name):
    s = str(name).strip().upper()
    s = re.sub(r"\b(EEG|REF|LE|RE|AVG|AVERAGE|CAR)\b", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    aliases = {"T7": "T3", "T3": "T3", "T8": "T4", "T4": "T4",
               "CZ": "CZ", "FCZ": "FCZ"}
    return aliases.get(s, s)


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
            f"Missing required electrodes {missing}. EDF channels are: "
            f"{list(ch_names)}. Check channel labels/aliases before proceeding."
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
    sos = signal.butter(
        N=4, Wn=[low, high], btype="bandpass", fs=fs, output="sos"
    )
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
    return signal.resample_poly(
        data, target_fs_i // g, orig_fs_i // g, axis=1
    )


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
        electrode_data[a] - electrode_data[b] for a, b in BIPOLAR_PAIRS
    ], axis=0) * 1e6

    bipolar = apply_filter(bipolar, original_fs)
    if abs(original_fs - target_fs) > 1e-6:
        bipolar = resample_signal(bipolar, original_fs, target_fs)

    n_keep = min(
        bipolar.shape[1],
        int(round(target_fs * 60 * RECORDING_MINUTES))
    )
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


# ---------------------------------------------------------------------
# DATASET — multi-step-ahead prediction
# ---------------------------------------------------------------------
class EEGWindowDataset(Dataset):
    """x: [SEQ_LEN, n_ch]; y: [n_ch] at t + HORIZON steps."""
    def __init__(self, recordings, seq_len=SEQ_LEN, horizon=HORIZON,
                 max_windows=None, rng=None):
        xs, ys = [], []
        stride = seq_len
        for rec in recordings:
            n_ch, T = rec.shape
            last_start = T - seq_len - horizon
            for start in range(0, last_start + 1, stride):
                xs.append(
                    rec[:, start:start + seq_len].T.astype(np.float32)
                )
                ys.append(
                    rec[:, start + seq_len + horizon - 1].astype(np.float32)
                )

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


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
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
    def __init__(self, input_size, hidden_size, output_size,
                 num_compartments=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TractableDendriticCell(
            input_size, hidden_size, num_compartments
        )
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
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class DANUCell(nn.Module):
    """
    use_gate=False -> d_gated = d_tilde (no gating layer).
    use_wta=False  -> uniform mean pooling over compartments.
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
            self.W_g = nn.Linear(
                in_dim + hidden_size, hidden_size * num_compartments
            )
        self.lambda_I = nn.Parameter(
            torch.ones(num_compartments, hidden_size) * 0.1
        )
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
            G = torch.sigmoid(
                self.W_g(xh_dbar).view(B, self.K, self.H)
            )
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
    def __init__(self, input_size, hidden_size, output_size,
                 num_compartments=5, use_gate=True, use_wta=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DANUCell(
            input_size, hidden_size, num_compartments,
            use_gate=use_gate, use_wta=use_wta
        )
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


def build_models(n_ch):
    """Build all models in the same order as the supplied original script."""
    return [
        (FULL, DANU(n_ch, HIDDEN_SIZE, n_ch, 5,
                    use_gate=True, use_wta=True), 5e-5, 1.0),
        ("DANU-noGate", DANU(n_ch, HIDDEN_SIZE, n_ch, 5,
                             use_gate=False, use_wta=True), 1e-5, 1.0),
        ("DANU-noWTA", DANU(n_ch, HIDDEN_SIZE, n_ch, 5,
                            use_gate=True, use_wta=False), 5e-5, 1.0),
        ("Tractable", TractableDendriticRNN(
            n_ch, HIDDEN_SIZE, n_ch, 4), 5e-5, 1.0),
        ("GRU", GRUModel(n_ch, HIDDEN_SIZE, n_ch), 5e-5, 1.0),
        ("LSTM", LSTMModel(n_ch, HIDDEN_SIZE, n_ch), 5e-5, 1.0),
    ]


# ---------------------------------------------------------------------
# TRAINING — CURVES SAVED AFTER EVERY EPOCH; NO CHECKPOINT RESUMPTION
# ---------------------------------------------------------------------
def train_model(model, train_loader, val_loader, history_file=None,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=1e-4,
                clip=1.0, truncate_k=None, log_grad_norm=False):
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

    stopped = max_epochs
    for epoch in range(max_epochs):
        model.train()
        train_sum, n_train = 0.0, 0
        grad_sum, n_grad = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(_forward(x), y)
            if torch.isfinite(loss):
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                train_sum += loss.item(); n_train += 1
                if log_grad_norm:
                    grad_sum += float(gn); n_grad += 1
        sched.step()
        train_loss = train_sum / n_train if n_train else float("nan")
        hist["train_loss"].append(train_loss)
        if log_grad_norm:
            hist["grad_norm"].append(grad_sum / n_grad if n_grad else float("nan"))

        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = loss_fn(_forward(x), y)
                if torch.isfinite(loss):
                    val_sum += loss.item(); n_val += 1
        val_loss = val_sum / n_val if n_val else float("nan")
        hist["val_loss"].append(val_loss)

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            stopped = epoch + 1
        hist["best_val"] = best_val
        hist["stopped"] = stopped
        # Atomic per-epoch curve persistence: partial curves survive interruption.
        if history_file:
            tmp = history_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(hist, f)
            os.replace(tmp, history_file)
        print(f"\r      epoch {epoch + 1:03d}/{max_epochs} "
              f"train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}",
              end="", flush=True)
        if patience_ctr >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val"] = best_val
    hist["stopped"] = stopped
    if history_file:
        tmp = history_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(hist, f)
        os.replace(tmp, history_file)
    return hist


# ---------------------------------------------------------------------
# PER-GRADE EXPERIMENT
# ---------------------------------------------------------------------
def run_grade_experiment(grade, recordings, seeds=SEEDS, history_store=None):
    n_ch = recordings[0].shape[0]
    print(f"\n{'=' * 70}\n  Grade {grade} | {len(recordings)} subjects | "
          f"{n_ch} channels | {len(seeds)} seeds\n{'=' * 70}")
    if len(recordings) < 3:
        print(f"  [SKIP] Not enough subjects (need >= 3, got {len(recordings)})")
        return

    for i, seed in enumerate(seeds):
        existing = load_intermediate_results()
        pending = [m for m in MODELS if not result_exists(existing, grade, seed, m)]
        if not pending:
            print(f"  Seed {seed}: all {len(MODELS)} combinations already in CSV; skipping")
            continue
        set_all_seeds(seed)
        train_loader, val_loader = make_loaders(recordings, seed)
        print(f"\n  Seed {seed} | train={len(train_loader.dataset)} | val={len(val_loader.dataset)}")
        # Instantiate in the original order, preserving the original RNG setup.
        models_for_seed = build_models(n_ch)
        for name, model, lr, clip in models_for_seed:
            if result_exists(load_intermediate_results(), grade, seed, name):
                print(f"    [CSV SKIP] {name}: result already exists (curve may be absent)")
                continue
            set_all_seeds(seed)
            if i == 0:
                print(f"    {name}: {n_params(model):,} params")
            print(f"    Training {name}...", end="", flush=True)
            hfile = curve_path(grade, seed, name)
            h = train_model(model, train_loader, val_loader,
                            history_file=hfile, lr=lr, clip=clip)
            # Persist the completed metric only after training has finished.
            save_intermediate_result(grade, seed, name, h["best_val"])
            if history_store is not None and seed == seeds[0]:
                history_store.setdefault(name, []).append(h)
            print(f"  MSE={h['best_val']:.6f} (ep {h['stopped']})")



# ---------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------
def summarise(grade_df):
    """Summaries use all CSV rows; paired tests use shared seed IDs only."""
    vals = {}
    by_model = {}
    for name in MODELS:
        part = grade_df.loc[grade_df["model"] == name, ["seed", "val_mse"]]
        part = part.drop_duplicates("seed", keep="last").sort_values("seed")
        by_model[name] = part.set_index("seed")["val_mse"]
        vals[name] = part["val_mse"].astype(float).tolist()

    summary = {}
    for name in MODELS:
        arr = np.asarray(vals[name], dtype=float)
        summary[name] = {
            "mean": float(np.mean(arr)) if len(arr) else float("nan"),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else (0.0 if len(arr) == 1 else float("nan")),
            "n": int(len(arr)),
        }

    def paired(a, b):
        common = by_model[a].index.intersection(by_model[b].index)
        x = by_model[a].loc[common].to_numpy(dtype=float)
        y = by_model[b].loc[common].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        return x[ok], y[ok], len(x[ok])

    for other in BASELINES:
        full, base, n = paired(FULL, other)
        p = stats.ttest_rel(full, base).pvalue if n >= 2 else float("nan")
        improvement = np.mean((base - full) / np.where(base != 0, base, np.nan) * 100) if n else float("nan")
        summary[f"p_danu_vs_{other}"] = float(p)
        summary[f"imp_vs_{other}"] = float(improvement)
        summary[f"n_danu_vs_{other}"] = n

    for abl in ABLATIONS:
        ablation, full, n = paired(abl, FULL)
        p = stats.ttest_rel(ablation, full).pvalue if n >= 2 else float("nan")
        degradation = np.mean((ablation - full) / np.where(full != 0, full, np.nan) * 100) if n else float("nan")
        wins = int(np.sum(full < ablation)) if n else 0
        summary[f"p_{abl}_vs_full"] = float(p)
        summary[f"deg_{abl}"] = float(degradation)
        summary[f"fullwins_{abl}"] = wins
        summary[f"n_{abl}_vs_full"] = n
    return summary, vals


def sig_stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def print_summary_table(grade_results):
    grades = sorted(grade_results.keys())
    print("\n" + "=" * 90)
    print(
        "  RESULTS — Cork Neonatal EEG — Prediction MSE "
        "(CSV results; paired tests use shared seeds)"
    )
    print("=" * 90)
    for g in grades:
        s = grade_results[g]["summary"]
        print(f"\n  -- Grade {g} --")
        for model in MODELS:
            print(
                f"    {model:18s} MSE = "
                f"{s[model]['mean']:.6f} ± {s[model]['std']:.6f} (n={s[model]['n']})"
            )
        print("    Full DANU vs baselines (+ = DANU better):")
        for b in BASELINES:
            p = s[f"p_danu_vs_{b}"]
            print(
                f"      vs {b:10s} {s[f'imp_vs_{b}']:+.2f}%  "
                f"p = {p:.4f} {sig_stars(p)} (paired n={s[f'n_danu_vs_{b}']})"
            )
        print("    Ablation effect (+ = removing component makes MSE worse):")
        for a in ABLATIONS:
            p = s[f"p_{a}_vs_full"]
            print(
                f"      {a:18s} {s[f'deg_{a}']:+.2f}%  "
                f"p = {p:.4f} {sig_stars(p)}  "
                f"full DANU better in {s[f'fullwins_{a}']}/{s[f'n_{a}_vs_full']} paired seeds"
            )

    print("\n" + "=" * 90)
    print("  ABLATION SUMMARY (mean over grades)")
    print("=" * 90)
    for a in ABLATIONS:
        d = np.nanmean([grade_results[g]["summary"][f"deg_{a}"] for g in grades])
        verdict = "component HELPS" if d > 0 else "component does NOT help"
        print(f"  {a:18s} mean degradation vs full = {d:+.2f}%  -> {verdict}")


COLORS = {
    "DANU": "#1B4332",
    "DANU-noGate": "#52B788",
    "DANU-noWTA": "#95D5B2",
    "Tractable": "#F18F01",
    "GRU": "#06A77D",
    "LSTM": "#A23B72",
}


def plot_grade_comparison(grade_results, save_path=None):
    grades = sorted(grade_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "DANU ablation (gating / WTA) vs baselines — Cork Neonatal EEG\n"
        "Prediction MSE by pathology grade (200 Hz), identical seeds and splits",
        fontsize=14, fontweight="bold"
    )
    ax = axes[0]
    x = np.arange(len(grades))
    width = 0.12
    offsets = (
        np.linspace(-(len(MODELS) - 1) / 2, (len(MODELS) - 1) / 2, len(MODELS))
        * width
    )
    for i, model in enumerate(MODELS):
        means = [grade_results[g]["summary"][model]["mean"] for g in grades]
        stds = [grade_results[g]["summary"][model]["std"] for g in grades]
        ax.bar(
            x + offsets[i], means, width, yerr=stds, capsize=3,
            color=COLORS[model], alpha=0.9, label=model
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Grade {g}" for g in grades], fontsize=11)
    ax.set_ylabel("Prediction MSE (lower = better)")
    ax.set_title("Prediction error by grade", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    xa = np.arange(len(grades))
    w = 0.3
    shift = (len(ABLATIONS) - 1) / 2
    for i, a in enumerate(ABLATIONS):
        degs = [grade_results[g]["summary"][f"deg_{a}"] for g in grades]
        ax2.bar(xa + (i - shift) * w, degs, w, color=COLORS[a], label=a)
        for gi, g in enumerate(grades):
            p = grade_results[g]["summary"][f"p_{a}_vs_full"]
            ax2.text(
                xa[gi] + (i - shift) * w, degs[gi], sig_stars(p),
                ha="center", va="bottom", fontsize=9
            )
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_xticks(xa)
    ax2.set_xticklabels([f"Grade {g}" for g in grades])
    ax2.set_ylabel(
        "MSE change vs full DANU (%)\n(+ = removing component hurts)"
    )
    ax2.set_title("Ablation effect", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot] Saved -> {save_path}")
    plt.close()


def plot_training_curves(grade_histories, save_path=None):
    grades = sorted(grade_histories.keys())
    n = len(grades)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.suptitle(
        "Training curves (seed 0): solid = val, dashed = train",
        fontsize=13, fontweight="bold"
    )
    for gi, grade in enumerate(grades):
        ax = axes[0, gi]
        for name, hist_list in grade_histories[grade].items():
            if not hist_list:
                continue
            h = hist_list[0]
            epochs = np.arange(1, len(h["val_loss"]) + 1)
            ax.plot(
                epochs, h["val_loss"], color=COLORS.get(name),
                linewidth=2, label=f"{name} (val)"
            )
            ax.plot(
                epochs, h["train_loss"], color=COLORS.get(name),
                linewidth=1, linestyle="--", alpha=0.6
            )
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


def save_csv(results_df):
    out = results_df.sort_values(["grade", "seed", "model"])
    out.to_csv(os.path.join(SAVE_DIR, "per_seed_mse.csv"), index=False)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  DANU Cork EEG — CSV-driven gating / WTA ablation")
    print(f"  fs={TARGET_FS} Hz | SEQ_LEN={SEQ_LEN} | HORIZON={HORIZON} | seeds={len(SEEDS)}")
    print(f"  models: {MODELS}\n  device: {DEVICE}")
    print(f"  Existing CSV results are authoritative: {results_path()}")
    print(f"  New training curves: {HISTORY_DIR}/")
    print("  Checkpoint files are ignored.")
    print("=" * 70)

    existing = load_intermediate_results()
    print(f"[CSV] Loaded {len(existing)} existing result rows; these combinations will not be retrained.")
    pending_grades = [
        grade for grade in GRADES
        if any(not result_exists(existing, grade, seed, model)
               for seed in SEEDS for model in MODELS)
    ]
    if pending_grades:
        print(f"[Data] Loading EEG data only for grades with missing CSV combinations: {pending_grades}")
        grade_data = load_subjects(DATA_DIR, METADATA_FILE,
                                   target_fs=TARGET_FS, grades=pending_grades)
        for grade in pending_grades:
            recordings = grade_data.get(grade, [])
            if len(recordings) < 3:
                print(f"[WARN] Grade {grade}: insufficient usable recordings ({len(recordings)}); no new runs")
                continue
            run_grade_experiment(grade, recordings, seeds=SEEDS)
    else:
        print("[Data] All requested grade/seed/model combinations are already in the CSV; skipping EDF loading and training.")

    # Re-read after training so summaries combine historical and newly completed runs.
    results_df = load_intermediate_results()
    results_df = results_df[results_df["grade"].isin(GRADES) &
                            results_df["seed"].isin(SEEDS) &
                            results_df["model"].isin(MODELS)].copy()
    grade_results, grade_histories = {}, {}
    for grade in GRADES:
        grade_df = results_df[results_df["grade"] == grade].copy()
        if grade_df.empty:
            continue
        summary, vals = summarise(grade_df)
        metrics = []
        for seed in sorted(grade_df["seed"].unique()):
            row = {"__seed": int(seed)}
            for model in MODELS:
                hit = grade_df[(grade_df["seed"] == seed) & (grade_df["model"] == model)]
                if not hit.empty:
                    row[model] = float(hit.iloc[-1]["val_mse"])
            metrics.append(row)
        grade_results[grade] = {"metrics": metrics, "summary": summary, "vals": vals}
        curves = load_seed0_curves(grade)
        if curves:
            grade_histories[grade] = curves

    if not grade_results:
        print("[ERROR] No matching results found in intermediate_results.csv.")
        return

    print_summary_table(grade_results)
    save_csv(results_df)
    plot_grade_comparison(grade_results,
        os.path.join(SAVE_DIR, "cork_danu_ablation_comparison.png"))
    if grade_histories:
        plot_training_curves(grade_histories,
            os.path.join(SAVE_DIR, "cork_danu_ablation_training_curves.png"))
    else:
        print("[Curves] No saved seed-0 curves found; no curve plot generated. Existing CSV rows were not retrained.")
    print(f"\n[Done] Results saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
