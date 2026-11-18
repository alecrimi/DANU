"""Cork EEG baselines (Tractable / GRU / LSTM), parameter-matched to DANU.

Same pipeline as cork_danu_ablation_patient_split.py:
  * same preprocessing, same patient-level split (same seeds -> same splits),
    same windows, same training loop / lr / batch / patience / clip,
  * CSV-driven experiment skipping (intermediate_results.csv is authoritative).

Differences:
  * only 3 models: Tractable (K=4), GRU, LSTM
  * hidden sizes are NOT fixed to 128: each model's hidden size is searched so
    that its trainable parameter count is as close as possible to TARGET_PARAMS
    (~170k, i.e. the size of the simplest DANU with K=4 compartments).
  * SAVE_DIR is new: results are not comparable with the 128-unit runs.
"""

import os
import re
import json
import random
import warnings
from math import gcd
from itertools import combinations

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
GRADES = [4]
SEQ_LEN = 512
HORIZON = 300
TARGET_PARAMS = 170_000      # parameter budget every model is matched to
TRACTABLE_K = 4              # compartments of the tractable dendritic RNN
MAX_EPOCHS = 400
PATIENCE = 25
BATCH_SIZE = 64
LR = 1e-4            # identical for every model (same as the DANU script)
CLIP = 1.0
VAL_RATIO = 0.2      # fraction of PATIENTS (not epochs) held out
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "cork_baselines_param_matched_results"
HISTORY_DIR = os.path.join(SAVE_DIR, "training_curves")
SPLIT_DIR = os.path.join(SAVE_DIR, "splits")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

MODELS = ["Tractable", "GRU", "LSTM"]

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
N_CH = len(BIPOLAR_PAIRS)


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
# PATIENT ID HANDLING
# ---------------------------------------------------------------------
def patient_id_from_file_id(file_id):
    """'ID43_epoch1' / 'ID43_epoch12' / 'id43-epoch2' -> 'ID43'.

    If no '_epochN' suffix is present the whole file_id is treated as the
    patient ID (one file per patient).
    """
    s = str(file_id).strip()
    s = re.sub(r"\.edf$", "", s, flags=re.IGNORECASE)
    m = re.match(r"^(.*?)[_\-\s]*epoch[_\-\s]*\d+$", s, flags=re.IGNORECASE)
    return (m.group(1) if m else s).strip().upper()


# ---------------------------------------------------------------------
# EDF LOADING + PREPROCESSING (unchanged)
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
    """Returns {grade: [(patient_id, recording[n_ch, T]), ...]}."""
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
        if "patient_id" in row.index and pd.notna(row["patient_id"]):
            patient_id = str(row["patient_id"]).strip().upper()
        else:
            patient_id = patient_id_from_file_id(file_id)
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
            grade_data.setdefault(grade, []).append((patient_id, processed))
            qc_rows.append({
                "file_id": file_id,
                "patient_id": patient_id,
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
    print(f"[Data] Loaded {loaded} recordings, skipped {skipped}")
    for g, recs in grade_data.items():
        n_pat = len({pid for pid, _ in recs})
        print(f"  Grade {g}: {len(recs)} recordings from {n_pat} patients")
    return grade_data


def n_unique_patients(recordings):
    return len({pid for pid, _ in recordings})


# ---------------------------------------------------------------------
# DATASET — multi-step-ahead prediction (unchanged)
# ---------------------------------------------------------------------
class EEGWindowDataset(Dataset):
    """x: [SEQ_LEN, n_ch]; y: [n_ch] at t + HORIZON steps. Uses ALL windows."""
    def __init__(self, recordings, seq_len=SEQ_LEN, horizon=HORIZON):
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
        self.x = torch.from_numpy(np.stack(xs))
        self.y = torch.from_numpy(np.stack(ys))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def split_by_patient(recordings, seed, val_ratio=VAL_RATIO):
    """Patient-level split: all recordings of a patient stay on one side."""
    patients = sorted({pid for pid, _ in recordings})
    if len(patients) < 2:
        raise ValueError("Need at least 2 patients for a patient-level split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(patients))
    n_val = int(len(patients) * val_ratio)
    n_val = min(max(1, n_val), len(patients) - 1)
    val_patients = {patients[i] for i in perm[:n_val]}
    train_patients = {patients[i] for i in perm[n_val:]}
    assert not (val_patients & train_patients), "Patient leakage in split!"

    train_recs = [rec for pid, rec in recordings if pid in train_patients]
    val_recs = [rec for pid, rec in recordings if pid in val_patients]
    return train_recs, val_recs, sorted(train_patients), sorted(val_patients)


def make_loaders(recordings, seed, grade=None, val_ratio=VAL_RATIO,
                 horizon=HORIZON):
    train_recs, val_recs, train_pat, val_pat = split_by_patient(
        recordings, seed, val_ratio
    )
    if grade is not None:  # audit trail of the split
        with open(os.path.join(SPLIT_DIR, f"grade{grade}_seed{seed}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"train_patients": train_pat,
                       "val_patients": val_pat,
                       "n_train_recordings": len(train_recs),
                       "n_val_recordings": len(val_recs)}, f, indent=2)

    train_ds = EEGWindowDataset(train_recs, horizon=horizon)
    val_ds = EEGWindowDataset(val_recs, horizon=horizon)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )
    return train_loader, val_loader, train_pat, val_pat


# ---------------------------------------------------------------------
# MODELS (same architectures as before; hidden size is now a free knob)
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
                 num_compartments=4):
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


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------
# PARAMETER MATCHING
# ---------------------------------------------------------------------
MODEL_BUILDERS = {
    "Tractable": lambda h, n_ch: TractableDendriticRNN(
        n_ch, h, n_ch, TRACTABLE_K),
    "GRU": lambda h, n_ch: GRUModel(n_ch, h, n_ch),
    "LSTM": lambda h, n_ch: LSTMModel(n_ch, h, n_ch),
}
HIDDEN_STEP = {"Tractable": TRACTABLE_K, "GRU": 1, "LSTM": 1}  # divisibility


def find_hidden_size(name, n_ch, target=TARGET_PARAMS, max_hidden=2048):
    """Hidden size whose parameter count is closest to `target`."""
    step = HIDDEN_STEP[name]
    best_h, best_diff, best_n = None, float("inf"), 0
    for h in range(step, max_hidden + 1, step):
        n = n_params(MODEL_BUILDERS[name](h, n_ch))
        diff = abs(n - target)
        if diff < best_diff:
            best_h, best_diff, best_n = h, diff, n
        if n > target and diff > best_diff:   # past the optimum
            break
    return best_h, best_n


def resolve_hidden_sizes(n_ch, target=TARGET_PARAMS):
    hidden, counts = {}, {}
    for name in MODELS:
        hidden[name], counts[name] = find_hidden_size(name, n_ch, target)
    return hidden, counts


HIDDEN, PARAM_COUNTS = resolve_hidden_sizes(N_CH)


def build_models(n_ch):
    """All models share the same lr (LR) and grad clip (CLIP)."""
    return [(name, MODEL_BUILDERS[name](HIDDEN[name], n_ch), LR, CLIP)
            for name in MODELS]


# ---------------------------------------------------------------------
# TRAINING — CURVES SAVED AFTER EVERY EPOCH (unchanged)
# ---------------------------------------------------------------------
def _dump_history(history_file, hist):
    tmp = history_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(hist, f)
    os.replace(tmp, history_file)


def train_model(model, train_loader, val_loader, history_file=None,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR,
                clip=CLIP, truncate_k=None, log_grad_norm=True):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs, eta_min=lr / 50
    )
    loss_fn = nn.MSELoss()
    best_val, patience_ctr, best_state, best_epoch = float("inf"), 0, None, 0
    hist = {"lr": lr, "batch_size": BATCH_SIZE, "patience": patience,
            "n_params": n_params(model),
            "stopped": max_epochs, "best_val": float("inf"), "best_epoch": 0,
            "train_loss": [], "val_loss": [], "lr_curve": []}
    if log_grad_norm:
        hist["grad_norm"] = []

    import inspect
    accepts_trunc = "truncate_k" in inspect.signature(model.forward).parameters

    def _forward(x):
        if accepts_trunc:
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
        hist["lr_curve"].append(opt.param_groups[0]["lr"])
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
            best_epoch = epoch + 1
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            stopped = epoch + 1
        hist["best_val"] = best_val
        hist["best_epoch"] = best_epoch
        hist["stopped"] = stopped
        if history_file:
            _dump_history(history_file, hist)
        print(f"\r      epoch {epoch + 1:03d}/{max_epochs} "
              f"train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}"
              f"@{best_epoch}", end="", flush=True)
        if patience_ctr >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val"] = best_val
    hist["best_epoch"] = best_epoch
    hist["stopped"] = stopped
    if history_file:
        _dump_history(history_file, hist)
    return hist


# ---------------------------------------------------------------------
# PER-GRADE EXPERIMENT
# ---------------------------------------------------------------------
def run_grade_experiment(grade, recordings, seeds=SEEDS, history_store=None):
    n_ch = recordings[0][1].shape[0]
    n_pat = n_unique_patients(recordings)
    print(f"\n{'=' * 70}\n  Grade {grade} | {len(recordings)} recordings | "
          f"{n_pat} patients | {n_ch} channels | {len(seeds)} seeds\n{'=' * 70}")
    if n_pat < 3:
        print(f"  [SKIP] Not enough patients (need >= 3, got {n_pat})")
        return

    for i, seed in enumerate(seeds):
        existing = load_intermediate_results()
        pending = [m for m in MODELS if not result_exists(existing, grade, seed, m)]
        if not pending:
            print(f"  Seed {seed}: all {len(MODELS)} combinations already in CSV; skipping")
            continue
        set_all_seeds(seed)
        train_loader, val_loader, train_pat, val_pat = make_loaders(
            recordings, seed, grade=grade
        )
        print(f"\n  Seed {seed} | train: {len(train_pat)} patients / "
              f"{len(train_loader.dataset)} windows | "
              f"val: {len(val_pat)} patients / {len(val_loader.dataset)} windows")
        models_for_seed = build_models(n_ch)
        for name, model, lr, clip in models_for_seed:
            if result_exists(load_intermediate_results(), grade, seed, name):
                print(f"    [CSV SKIP] {name}: result already exists")
                continue
            set_all_seeds(seed)
            if i == 0:
                print(f"    {name}: hidden={HIDDEN[name]} | "
                      f"{n_params(model):,} params")
            print(f"    Training {name}...", end="", flush=True)
            hfile = curve_path(grade, seed, name)
            h = train_model(model, train_loader, val_loader,
                            history_file=hfile, lr=lr, clip=clip)
            save_intermediate_result(grade, seed, name, h["best_val"])
            if history_store is not None and seed == seeds[0]:
                history_store.setdefault(name, []).append(h)
            print(f"  MSE={h['best_val']:.6f} (best ep {h['best_epoch']}, "
                  f"stopped ep {h['stopped']})")


# ---------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------
PAIRS = list(combinations(MODELS, 2))   # every model pair


def summarise(grade_df):
    """Summaries use all CSV rows; paired tests use shared seed IDs only."""
    vals, by_model = {}, {}
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

    for a, b in PAIRS:
        common = by_model[a].index.intersection(by_model[b].index)
        x = by_model[a].loc[common].to_numpy(dtype=float)
        y = by_model[b].loc[common].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        n = len(x)
        p = stats.ttest_rel(x, y).pvalue if n >= 2 else float("nan")
        # + = a better (lower MSE) than b
        imp = (float(np.mean((y - x) / np.where(y != 0, y, np.nan) * 100))
               if n else float("nan"))
        summary[f"p_{a}_vs_{b}"] = float(p)
        summary[f"imp_{a}_vs_{b}"] = imp
        summary[f"n_{a}_vs_{b}"] = n
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
    print("  RESULTS — Cork Neonatal EEG — Prediction MSE "
          f"(~{TARGET_PARAMS:,} params per model; patient-level split)")
    print("=" * 90)
    for model in MODELS:
        print(f"  {model:10s} hidden={HIDDEN[model]:4d}  params={PARAM_COUNTS[model]:,}")
    for g in grades:
        s = grade_results[g]["summary"]
        print(f"\n  -- Grade {g} --")
        for model in MODELS:
            print(f"    {model:10s} MSE = {s[model]['mean']:.6f} "
                  f"± {s[model]['std']:.6f} (n={s[model]['n']})")
        print("    Pairwise (+ = first model better, paired t-test):")
        for a, b in PAIRS:
            p = s[f"p_{a}_vs_{b}"]
            print(f"      {a:10s} vs {b:10s} {s[f'imp_{a}_vs_{b}']:+.2f}%  "
                  f"p = {p:.4f} {sig_stars(p)} (n={s[f'n_{a}_vs_{b}']})")


COLORS = {"Tractable": "#F18F01", "GRU": "#06A77D", "LSTM": "#A23B72"}


def plot_grade_comparison(grade_results, save_path=None):
    grades = sorted(grade_results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Parameter-matched baselines (~{TARGET_PARAMS // 1000}k params) — "
        "Cork Neonatal EEG\n"
        f"Prediction MSE by grade, patient-level split, lr={LR:g}, "
        f"batch={BATCH_SIZE}, patience={PATIENCE}",
        fontsize=13, fontweight="bold"
    )
    x = np.arange(len(grades))
    width = 0.25
    offsets = np.linspace(-(len(MODELS) - 1) / 2, (len(MODELS) - 1) / 2,
                          len(MODELS)) * width
    for i, model in enumerate(MODELS):
        means = [grade_results[g]["summary"][model]["mean"] for g in grades]
        stds = [grade_results[g]["summary"][model]["std"] for g in grades]
        ax.bar(x + offsets[i], means, width, yerr=stds, capsize=3,
               color=COLORS[model], alpha=0.9,
               label=f"{model} (h={HIDDEN[model]}, {PARAM_COUNTS[model]:,} p)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Grade {g}" for g in grades], fontsize=11)
    ax.set_ylabel("Prediction MSE (lower = better)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot] Saved -> {save_path}")
    plt.close()


def plot_training_curves(grade_histories, save_path=None):
    """Seed-0 curves: solid = val, dashed = train. Best epoch marked with a dot."""
    grades = sorted(grade_histories.keys())
    n = len(grades)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.suptitle(
        f"Training curves (seed 0, lr={LR:g}): solid = val, dashed = train, "
        "dot = best epoch",
        fontsize=13, fontweight="bold"
    )
    for gi, grade in enumerate(grades):
        ax = axes[0, gi]
        for name, hist_list in grade_histories[grade].items():
            if not hist_list:
                continue
            h = hist_list[0]
            epochs = np.arange(1, len(h["val_loss"]) + 1)
            ax.plot(epochs, h["val_loss"], color=COLORS.get(name),
                    linewidth=2, label=f"{name} (val)")
            ax.plot(epochs, h["train_loss"], color=COLORS.get(name),
                    linewidth=1, linestyle="--", alpha=0.6)
            be = h.get("best_epoch", 0)
            if 1 <= be <= len(h["val_loss"]):
                ax.plot(be, h["val_loss"][be - 1], "o",
                        color=COLORS.get(name), markersize=6)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (log scale)")
        ax.set_title(f"Grade {grade}", fontweight="bold")
        ax.grid(alpha=0.3)
        if gi == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Plot] Saved -> {save_path}")
    plt.close()


def save_csv(results_df):
    out = results_df.sort_values(["grade", "seed", "model"])
    out.to_csv(os.path.join(SAVE_DIR, "per_seed_mse.csv"), index=False)


def save_param_table():
    pd.DataFrame([{"model": m, "hidden_size": HIDDEN[m],
                   "n_params": PARAM_COUNTS[m], "target": TARGET_PARAMS}
                  for m in MODELS]
                 ).to_csv(os.path.join(SAVE_DIR, "param_counts.csv"), index=False)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Cork EEG — parameter-matched baselines (Tractable / GRU / LSTM)")
    print(f"  fs={TARGET_FS} Hz | SEQ_LEN={SEQ_LEN} | HORIZON={HORIZON} | seeds={len(SEEDS)}")
    print(f"  lr={LR:g} (all models) | batch={BATCH_SIZE} | patience={PATIENCE} "
          f"| max_epochs={MAX_EPOCHS} | val_ratio={VAL_RATIO} (patients)")
    print(f"  target params ~{TARGET_PARAMS:,}")
    for m in MODELS:
        print(f"    {m:10s} hidden={HIDDEN[m]:4d}  params={PARAM_COUNTS[m]:,}")
    print(f"  device: {DEVICE}")
    print(f"  Results CSV (authoritative): {results_path()}")
    print("=" * 70)
    save_param_table()

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
            n_pat = n_unique_patients(recordings)
            if n_pat < 3:
                print(f"[WARN] Grade {grade}: insufficient patients ({n_pat}); no new runs")
                continue
            run_grade_experiment(grade, recordings, seeds=SEEDS)
    else:
        print("[Data] All requested grade/seed/model combinations are already in the CSV; skipping EDF loading and training.")

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
        grade_results[grade] = {"summary": summary, "vals": vals}
        curves = load_seed0_curves(grade)
        if curves:
            grade_histories[grade] = curves

    if not grade_results:
        print("[ERROR] No matching results found in intermediate_results.csv.")
        return

    print_summary_table(grade_results)
    save_csv(results_df)
    plot_grade_comparison(grade_results,
        os.path.join(SAVE_DIR, "cork_baselines_param_matched_comparison.png"))
    if grade_histories:
        plot_training_curves(grade_histories,
            os.path.join(SAVE_DIR, "cork_baselines_param_matched_training_curves.png"))
    else:
        print("[Curves] No saved seed-0 curves found; no curve plot generated.")
    print(f"\n[Done] Results saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
