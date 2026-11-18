
#!/usr/bin/env python3
"""
Lorenz-96 forecasting: DANU-4 vs LSTM vs GRU vs Tractable-RNN.

Task:
    Input:  100 historical time points
    Output: One 8-dimensional future state at horizon 10

Architectures:
    1. DANU-4
    2. LSTM
    3. GRU
    4. Tractable Dendritic RNN

Experimental design:
    - Identical datasets for all models within each seed
    - 5 random seeds
    - Adam optimizer
    - Cosine annealing learning rate
    - Early stopping
    - Gradient clipping
    - Paired statistical comparisons
    - Per-seed checkpoints
"""

import os
import json
import pickle
import time
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from scipy import stats
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_SAVE_DIR = "lorenz96_architecture_comparison"
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)


def drive_path(filename):
    return os.path.join(DRIVE_SAVE_DIR, filename)


def save_seed_checkpoint(
    seed_idx, seed, model_hists, metrics, dataset_name
):
    checkpoint = {
        "seed_idx": seed_idx,
        "seed": seed,
        "model_hists": model_hists,
        "metrics": metrics,
        "dataset_name": dataset_name,
        "timestamp": time.time(),
        "models": [
            "DANU-4", "LSTM", "GRU", "Tractable-RNN"
        ],
    }

    fname = drive_path(
        f"seed_{seed_idx:02d}_seed{seed}.pkl"
    )

    with open(fname, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"  Checkpoint saved: {fname}")


def load_seed_checkpoint(seed_idx, seed):
    fname = drive_path(
        f"seed_{seed_idx:02d}_seed{seed}.pkl"
    )

    if os.path.exists(fname):
        with open(fname, "rb") as f:
            cp = pickle.load(f)

        expected = {
            "DANU-4", "LSTM", "GRU", "Tractable-RNN"
        }

        if expected.issubset(set(cp.get("models", []))):
            print(f"  Resuming checkpoint: {fname}")
            return cp

        print("  Old/incompatible checkpoint. Retraining seed.")

    return None


def save_final_results(results, dataset_name):
    tag = dataset_name.replace(" ", "_").replace("/", "-")

    fname = drive_path(f"final_results_{tag}.pkl")

    with open(fname, "wb") as f:
        pickle.dump(results, f)

    summary = {
        k: (
            float(v)
            if isinstance(v, (np.floating, float))
            else v
        )
        for k, v in results.get("statistics", {}).items()
    }

    summary_fname = drive_path(
        f"final_summary_{tag}.json"
    )

    with open(summary_fname, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFinal results: {fname}")
    print(f"Final summary: {summary_fname}")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_EPOCHS = 300
PATIENCE = 50
BATCH_SIZE = 32
HIDDEN_SIZE = 128

NUM_SEEDS = 5
SEEDS = list(range(NUM_SEEDS))

LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 0.5

# Lorenz-96
L96_N = 8
L96_F = 8
L96_DT = 0.05

SEQ_LEN = 100
HORIZON = 10
TRAJ_LEN = 50000

NUM_COMPARTMENTS = 4

MODEL_NAMES = [
    "DANU-4",
    "LSTM",
    "GRU",
    "Tractable-RNN",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LORENZ-96 GENERATOR: RK4
# ============================================================

def lorenz96_rhs(state, F=L96_F):

    return (
        (np.roll(state, -1) - np.roll(state, 2))
        * np.roll(state, 1)
        - state
        + F
    )


def lorenz96_step(state, F=L96_F, dt=L96_DT):

    k1 = lorenz96_rhs(state, F=F)

    k2 = lorenz96_rhs(
        state + dt / 2 * k1, F=F
    )

    k3 = lorenz96_rhs(
        state + dt / 2 * k2, F=F
    )

    k4 = lorenz96_rhs(
        state + dt * k3, F=F
    )

    return state + (dt / 6) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )


def generate_lorenz96(
    T=TRAJ_LEN,
    N=L96_N,
    F=L96_F,
    dt=L96_DT,
    warmup=2000
):

    state = F * np.ones(N)
    state[0] += 0.01

    for _ in range(warmup):

        state = lorenz96_step(
            state, F=F, dt=dt
        )

        if not np.isfinite(state).all():
            raise ValueError(
                "Lorenz-96 diverged during warmup."
            )

    traj = []

    for _ in range(T):

        state = lorenz96_step(
            state, F=F, dt=dt
        )

        traj.append(state.copy())

    traj = np.array(traj)

    if not np.isfinite(traj).all():
        raise ValueError(
            "Lorenz-96 trajectory contains NaN/Inf."
        )

    # Normalize using the entire generated trajectory.
    traj = (
        traj - traj.mean(axis=0)
    ) / (
        traj.std(axis=0) + 1e-8
    )

    return traj


# ============================================================
# DATASET
# ============================================================

class Lorenz96Dataset(Dataset):

    def __init__(
        self,
        traj,
        indices,
        seq_len=SEQ_LEN,
        prediction_horizon=HORIZON
    ):

        self.data = []
        self.targets = []

        for i in indices:

            x = traj[i:i + seq_len]

            # Single future state, not a sequence.
            y = traj[
                i + seq_len + prediction_horizon - 1
            ]

            self.data.append(x)
            self.targets.append(y)

        self.data = torch.tensor(
            np.array(self.data),
            dtype=torch.float32
        )

        self.targets = torch.tensor(
            np.array(self.targets),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def make_train_val_datasets(
    seed,
    seq_len=SEQ_LEN,
    prediction_horizon=HORIZON,
    N=L96_N,
    F=L96_F,
    n_train=1200,
    n_val=300
):

    # One trajectory per seed, shared by all architectures.
    set_seed(seed)

    traj = generate_lorenz96(
        T=TRAJ_LEN,
        N=N,
        F=F
    )

    max_start = (
        len(traj) - seq_len - prediction_horizon
    )

    stride = max(1, seq_len // 4)

    all_starts = list(
        range(0, max_start, stride)
    )

    split = int(0.8 * len(all_starts))

    train_pool = all_starts[:split]

    last_train_end = (
        train_pool[-1] + seq_len + prediction_horizon
    )

    first_val_start = last_train_end + seq_len

    val_pool = [
        s for s in all_starts[split:]
        if s >= first_val_start
    ]

    if len(train_pool) < n_train:
        raise ValueError("Insufficient training windows.")

    if len(val_pool) < n_val:
        raise ValueError("Insufficient validation windows.")

    # Deterministic sampling per seed.
    rng = np.random.default_rng(seed)

    train_idx = rng.permutation(
        train_pool
    )[:n_train]

    val_idx = rng.permutation(
        val_pool
    )[:n_val]

    train_ds = Lorenz96Dataset(
        traj,
        train_idx,
        seq_len,
        prediction_horizon
    )

    val_ds = Lorenz96Dataset(
        traj,
        val_idx,
        seq_len,
        prediction_horizon
    )

    return train_ds, val_ds


# ============================================================
# DANU
# ============================================================

class DANUCell(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_compartments
    ):

        super().__init__()

        self.K = num_compartments
        self.H = hidden_size

        in_dim = input_size + hidden_size

        self.W_E = nn.Linear(
            in_dim,
            hidden_size * num_compartments
        )

        self.W_I = nn.Linear(
            in_dim,
            hidden_size * num_compartments
        )

        self.W_g = nn.Linear(
            in_dim + hidden_size,
            hidden_size * num_compartments
        )

        self.lambda_I = nn.Parameter(
            torch.ones(num_compartments, hidden_size) * 0.1
        )

        self.ln = nn.LayerNorm(hidden_size)

        self.beta = nn.Parameter(
            torch.ones(num_compartments, 1)
        )

        self.theta = nn.Parameter(
            torch.zeros(num_compartments, 1)
        )

        self.gamma = nn.Parameter(
            torch.ones(num_compartments, 1) * 0.1
        )

        self.W_q = nn.Linear(
            hidden_size, hidden_size
        )

        self.W_out = nn.Linear(
            hidden_size, hidden_size
        )

        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, x, h):

        B = x.size(0)

        xh = torch.cat([x, h], dim=1)

        E = self.W_E(xh).view(
            B, self.K, self.H
        )

        I = self.W_I(xh).view(
            B, self.K, self.H
        )

        u = E - self.lambda_I * I

        u = self.ln(u)

        sig = torch.sigmoid(
            self.beta * (u - self.theta)
        )

        nmda = (
            self.gamma * (u ** 2)
            / (1.0 + u ** 2)
        )

        d_tilde = sig + nmda

        d_bar = d_tilde.mean(
            dim=1, keepdim=True
        )

        xh_dbar = torch.cat(
            [xh, d_bar.squeeze(1)],
            dim=1
        )

        G = torch.sigmoid(
            self.W_g(xh_dbar).view(
                B, self.K, self.H
            )
        )

        d_gated = G * d_tilde

        q = self.W_q(h).unsqueeze(2)

        scores = torch.bmm(
            d_gated, q
        ).squeeze(2) / (self.H ** 0.5)

        alpha = torch.softmax(
            scores, dim=1
        )

        attended = (
            alpha.unsqueeze(2) * d_gated
        ).sum(dim=1)

        h_new = torch.tanh(
            self.ln_out(
                self.W_out(attended) + h
            )
        )

        return h_new


class DANU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_compartments
    ):

        super().__init__()

        self.hidden_size = hidden_size

        self.cell = DANUCell(
            input_size,
            hidden_size,
            num_compartments
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        B, T, _ = x.shape

        h = torch.zeros(
            B,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype
        )

        for t in range(T):
            h = self.cell(x[:, t], h)

        return self.fc(h)


# ============================================================
# TRACTABLE DENDRITIC RNN
# ============================================================

class TractableDendriticCell(nn.Module):
    """
    Passive dendritic compartments.

    - Static nonlinear compartments
    - No adaptive gating
    - No global modulation
    - No residual connection
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_compartments=4
    ):

        super().__init__()

        assert hidden_size % num_compartments == 0

        self.hidden_size = hidden_size
        self.num_comp = num_compartments

        self.comp_size = (
            hidden_size // num_compartments
        )

        self.W_in = nn.ModuleList([
            nn.Linear(
                input_size,
                self.comp_size
            )
            for _ in range(num_compartments)
        ])

        self.W_rec = nn.ModuleList([
            nn.Linear(
                hidden_size,
                self.comp_size
            )
            for _ in range(num_compartments)
        ])

        self.integration = nn.Linear(
            hidden_size,
            hidden_size
        )

    def forward(self, x, h):

        comp_outputs = []

        for i in range(self.num_comp):

            local_out = torch.tanh(
                self.W_in[i](x)
                + self.W_rec[i](h)
            )

            comp_outputs.append(local_out)

        combined = torch.cat(
            comp_outputs,
            dim=1
        )

        h_new = torch.tanh(
            self.integration(combined)
        )

        return h_new


class TractableDendriticRNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_compartments=4
    ):

        super().__init__()

        self.hidden_size = hidden_size

        self.cell = TractableDendriticCell(
            input_size,
            hidden_size,
            num_compartments
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        B, T, _ = x.shape

        h = torch.zeros(
            B,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype
        )

        for t in range(T):
            h = self.cell(x[:, t], h)

        return self.fc(h)


# ============================================================
# LSTM
# ============================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        _, (h_n, _) = self.rnn(x)

        return self.fc(h_n[-1])


# ============================================================
# GRU
# ============================================================

class GRUModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):

        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        _, h_n = self.rnn(x)

        return self.fc(h_n[-1])


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    name,
    input_size,
    hidden_size,
    output_size
):

    if name == "DANU-4":

        return DANU(
            input_size,
            hidden_size,
            output_size,
            num_compartments=NUM_COMPARTMENTS
        )

    elif name == "LSTM":

        return LSTMModel(
            input_size,
            hidden_size,
            output_size
        )

    elif name == "GRU":

        return GRUModel(
            input_size,
            hidden_size,
            output_size
        )

    elif name == "Tractable-RNN":

        return TractableDendriticRNN(
            input_size,
            hidden_size,
            output_size,
            num_compartments=NUM_COMPARTMENTS
        )

    else:
        raise ValueError(f"Unknown model: {name}")


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# TRAINING WITH EARLY STOPPING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    lr=LEARNING_RATE,
    device=DEVICE,
    clip_grad=True,
    max_grad_norm=MAX_GRAD_NORM
):

    model.to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max_epochs,
        eta_min=lr / 50
    )

    loss_fn = nn.MSELoss()

    hist = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val = float("inf")
    patience_counter = 0
    best_state = None
    stopped_epoch = max_epochs

    for epoch in range(max_epochs):

        model.train()

        tl = 0.0
        valid_batches = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()

            pred = model(x)

            loss = loss_fn(pred, y)

            if not torch.isfinite(loss):
                continue

            loss.backward()

            if clip_grad:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm
                )

            opt.step()

            tl += loss.item()
            valid_batches += 1

        scheduler.step()

        if valid_batches == 0:

            hist["train_loss"].append(float("nan"))
            hist["val_loss"].append(float("nan"))

            continue

        tl /= valid_batches

        # Validation
        model.eval()

        vl = 0.0
        valid_val_batches = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                loss = loss_fn(
                    model(x),
                    y
                )

                if torch.isfinite(loss):

                    vl += loss.item()
                    valid_val_batches += 1

        if valid_val_batches == 0:

            hist["train_loss"].append(tl)
            hist["val_loss"].append(float("nan"))

            continue

        vl /= valid_val_batches

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)

        if vl < best_val:

            best_val = vl
            patience_counter = 0

            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }

        else:

            patience_counter += 1

            if patience_counter >= patience:

                stopped_epoch = epoch + 1

                print(
                    f"    Early stop at epoch "
                    f"{stopped_epoch} "
                    f"(best val={best_val:.6f})"
                )

                break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist["stopped_epoch"] = stopped_epoch
    hist["best_val"] = best_val

    return hist


# ============================================================
# MULTI-MODEL EXPERIMENT
# ============================================================

def run_comparison(
    dataset_name,
    input_size,
    output_size,
    hidden_size=HIDDEN_SIZE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    seeds=SEEDS
):

    all_hists = {
        name: [] for name in MODEL_NAMES
    }

    final_metrics = []

    parameter_counts = {}

    print("\n" + "=" * 80)
    print(f"Experiment: {dataset_name}")
    print(f"Device: {device}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Seeds: {len(seeds)}")
    print("=" * 80)

    for idx, seed in enumerate(seeds):

        print(
            f"\n{'='*60}\n"
            f"Seed {idx+1}/{len(seeds)} (seed={seed})\n"
            f"{'='*60}"
        )

        cp = load_seed_checkpoint(idx, seed)

        if cp is not None:

            for name in MODEL_NAMES:
                all_hists[name].append(
                    cp["model_hists"][name]
                )

            final_metrics.append(cp["metrics"])

            if "parameter_counts" in cp:
                parameter_counts = cp["parameter_counts"]

            print("Loaded all four models from checkpoint.")

            continue

        # Identical data for all models within this seed.
        train_dataset, val_dataset = make_train_val_datasets(
            seed=seed,
            seq_len=SEQ_LEN,
            prediction_horizon=HORIZON,
            N=L96_N,
            F=L96_F,
            n_train=1200,
            n_val=300
        )

        # Identical batches/order for every architecture.
        train_generator = torch.Generator()
        train_generator.manual_seed(seed + 10000)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        seed_hists = {}
        seed_metrics = {}

        for model_idx, name in enumerate(MODEL_NAMES):

            print(f"\nTraining {name}...")

            # Independent initialization, reproducible per model/seed.
            set_seed(seed * 100 + model_idx)

            model = create_model(
                name,
                input_size,
                hidden_size,
                output_size
            ).to(device)

            n_params = count_parameters(model)

            parameter_counts[name] = n_params

            print(f"  Parameters: {n_params:,}")

            hist = train_model(
                model,
                train_loader,
                val_loader,
                max_epochs=max_epochs,
                patience=patience,
                lr=LEARNING_RATE,
                device=device,
                clip_grad=True,
                max_grad_norm=MAX_GRAD_NORM
            )

            seed_hists[name] = hist

            seed_metrics[f"{name}_best"] = hist["best_val"]

            seed_metrics[
                f"{name}_stopped_epoch"
            ] = hist["stopped_epoch"]

            print(
                f"  Best validation MSE: "
                f"{hist['best_val']:.6f}"
            )

            # Free GPU memory.
            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Checkpoint
        checkpoint = {
            "seed_idx": idx,
            "seed": seed,
            "model_hists": seed_hists,
            "metrics": seed_metrics,
            "dataset_name": dataset_name,
            "timestamp": time.time(),
            "models": MODEL_NAMES,
            "parameter_counts": parameter_counts,
        }

        fname = drive_path(
            f"seed_{idx:02d}_seed{seed}.pkl"
        )

        with open(fname, "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"\nCheckpoint saved: {fname}")

        for name in MODEL_NAMES:
            all_hists[name].append(seed_hists[name])

        final_metrics.append(seed_metrics)

    # ========================================================
    # AGGREGATION
    # ========================================================

    def pad_and_stack(hist_list, key):

        arrays = [
            np.array(h[key], dtype=float)
            for h in hist_list
        ]

        max_len = max(len(a) for a in arrays)

        padded = [
            np.pad(
                a,
                (0, max_len - len(a)),
                mode="edge"
            )
            for a in arrays
        ]

        return np.stack(padded, axis=0)

    results = {
        "final_metrics": final_metrics,
        "parameter_counts": parameter_counts,
        "model_names": MODEL_NAMES,
    }

    for name in MODEL_NAMES:

        tag = name.lower().replace("-", "")

        for split in ["train", "val"]:

            stacked = pad_and_stack(
                all_hists[name],
                f"{split}_loss"
            )

            results[f"{tag}_{split}_mean"] = (
                stacked.mean(axis=0)
            )

            results[f"{tag}_{split}_std"] = (
                stacked.std(axis=0)
            )

    # ========================================================
    # FINAL PERFORMANCE
    # ========================================================

    bests = {
        name: [
            m[f"{name}_best"]
            for m in final_metrics
        ]
        for name in MODEL_NAMES
    }

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print("\nValidation MSE (mean ± SD):")

    for name in MODEL_NAMES:

        values = bests[name]

        print(
            f"{name:18s}: "
            f"{np.mean(values):.6f} ± "
            f"{np.std(values):.6f}"
        )

    print("\nTrainable parameters:")

    for name in MODEL_NAMES:

        print(
            f"{name:18s}: "
            f"{parameter_counts.get(name, 0):,}"
        )

    print("\nConvergence epochs:")

    for name in MODEL_NAMES:

        epochs = [
            m[f"{name}_stopped_epoch"]
            for m in final_metrics
        ]

        print(
            f"{name:18s}: "
            f"{np.mean(epochs):.1f} ± "
            f"{np.std(epochs):.1f}"
        )

    # ========================================================
    # PAIRED STATISTICAL TESTS
    # ========================================================

    def significance(p):

        if p < 0.001:
            return "***"

        if p < 0.01:
            return "**"

        if p < 0.05:
            return "*"

        return "n.s."

    stats_out = {}

    print("\nPaired t-tests:")

    for i, name_a in enumerate(MODEL_NAMES):

        for name_b in MODEL_NAMES[i + 1:]:

            _, p = stats.ttest_rel(
                bests[name_a],
                bests[name_b]
            )

            key = f"p_{name_a}_vs_{name_b}"

            stats_out[key] = float(p)

            print(
                f"{name_a:18s} vs {name_b:18s}: "
                f"p={p:.6f} {significance(p)}"
            )

    results["statistics"] = stats_out

    save_final_results(
        results,
        dataset_name
    )

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_results(
    results,
    title="Validation Loss Across Seeds (Mean ± Std, first 250 epochs)",
    save_path=None
):
    """
    Plot validation loss averaged across the experimental seeds.

    Only the first 250 epochs are displayed. The figure title and
    legend describe the plotted quantities/models, without reporting
    the number of seeds/runs.
    """

    # Keep the requested legend order and display names.
    models = [
        ("danu4", "DANU", "#174735"),
        ("gru", "GRU", "#00A77B"),
        ("lstm", "LSTM", "#AD367B"),
        ("tractablernn", "Tractable", "#F28E00"),
    ]

    max_epochs_to_plot = 250

    fig, ax = plt.subplots(figsize=(12, 7))

    for key, label, color in models:
        mean = np.asarray(
            results[f"{key}_val_mean"], dtype=float
        )[:max_epochs_to_plot]
        std = np.asarray(
            results[f"{key}_val_std"], dtype=float
        )[:max_epochs_to_plot]

        epochs = np.arange(len(mean))

        # A log axis requires positive lower confidence-band values.
        lower = np.maximum(mean - std, 1e-8)
        upper = np.maximum(mean + std, 1e-8)

        ax.plot(
            epochs,
            mean,
            color=color,
            label=label,
            linewidth=2.5
        )
        ax.fill_between(
            epochs,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Validation Loss (log scale)", fontsize=13)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

    ax.set_xlim(0, max_epochs_to_plot)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=11)

    fig.tight_layout()

    if save_path:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )
        print(f"Validation-loss plot saved: {save_path}")

    plt.show()
    return fig, ax


# ============================================================
# RELOAD CHECKPOINTS
# ============================================================

def reload_all_checkpoints(seeds=SEEDS):

    all_hists = {
        name: [] for name in MODEL_NAMES
    }

    final_metrics = []

    for idx, seed in enumerate(seeds):

        cp = load_seed_checkpoint(idx, seed)

        if cp is None:

            raise FileNotFoundError(
                f"Missing complete checkpoint for seed {seed}"
            )

        for name in MODEL_NAMES:

            all_hists[name].append(
                cp["model_hists"][name]
            )

        final_metrics.append(cp["metrics"])

    def pad_and_stack(hist_list, key):

        arrays = [
            np.array(h[key])
            for h in hist_list
        ]

        max_len = max(
            len(a) for a in arrays
        )

        padded = [
            np.pad(
                a,
                (0, max_len - len(a)),
                mode="edge"
            )
            for a in arrays
        ]

        return np.stack(padded, axis=0)

    results = {
        "final_metrics": final_metrics,
        "model_names": MODEL_NAMES,
    }

    for name in MODEL_NAMES:

        tag = name.lower().replace("-", "")

        for split in ["train", "val"]:

            stacked = pad_and_stack(
                all_hists[name],
                f"{split}_loss"
            )

            results[f"{tag}_{split}_mean"] = (
                stacked.mean(axis=0)
            )

            results[f"{tag}_{split}_std"] = (
                stacked.std(axis=0)
            )

    results["parameter_counts"] = cp.get(
        "parameter_counts", {}
    )

    print(
        f"Reloaded {len(seeds)} seeds "
        f"from {DRIVE_SAVE_DIR}"
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)

    print(
        "Lorenz-96: DANU-4 vs LSTM vs GRU vs Tractable-RNN"
    )

    print(
        f"L96: N={L96_N}, F={L96_F}, dt={L96_DT}"
    )

    print(
        f"Task: seq_len={SEQ_LEN}, horizon={HORIZON}"
    )

    print(
        f"Hidden size: {HIDDEN_SIZE}"
    )

    print(
        f"Compartments: {NUM_COMPARTMENTS}"
    )

    print(
        f"Seeds: {NUM_SEEDS}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Saving to: {DRIVE_SAVE_DIR}"
    )

    print("=" * 80)

    DATASET_NAME = (
        f"Lorenz96_N{L96_N}_F{L96_F}"
        f"_seq{SEQ_LEN}_h{HORIZON}"
    )

    results = run_comparison(
        dataset_name=DATASET_NAME,
        input_size=L96_N,
        output_size=L96_N,
        hidden_size=HIDDEN_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        seeds=SEEDS
    )

    visualize_results(
        results,
        title="Validation Loss Across Seeds (Mean ± Std, first 250 epochs)",
        save_path=drive_path(
            "validation_loss_across_seeds.png"
        )
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {DRIVE_SAVE_DIR}")
    print("=" * 80)
