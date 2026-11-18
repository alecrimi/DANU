import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import stats
import os
import json
import pickle
import time

# ============================
# Google Drive setup
# ============================
DRIVE_SAVE_DIR = "lorenz96_danu_ablation"
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

def drive_path(filename):
    return os.path.join(DRIVE_SAVE_DIR, filename)

def save_seed_checkpoint(seed_idx, seed, model_hists, metrics, dataset_name):
    checkpoint = {
        "seed_idx":     seed_idx,
        "seed":         seed,
        "model_hists":  model_hists,
        "metrics":      metrics,
        "dataset_name": dataset_name,
        "timestamp":    time.time(),
    }
    fname = drive_path(f"seed_{seed_idx:02d}_seed{seed}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  ✓ Checkpoint saved → {fname}")
    return fname

def load_seed_checkpoint(seed_idx, seed):
    fname = drive_path(f"seed_{seed_idx:02d}_seed{seed}.pkl")
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            cp = pickle.load(f)
        print(f"  ↩ Resuming from checkpoint: {fname}")
        return cp
    return None

def save_final_results(results, dataset_name):
    tag = dataset_name.replace(" ", "_").replace("/", "-")
    fname = drive_path(f"final_results_{tag}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    summary = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
               for k, v in results.get("statistics", {}).items()}
    summary_fname = drive_path(f"final_summary_{tag}.json")
    with open(summary_fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Final results saved → {fname}")
    print(f"✓ Summary JSON saved  → {summary_fname}")

# ============================
# Global config
# ============================
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS  = 300
PATIENCE    = 50
BATCH_SIZE  = 32
HIDDEN_SIZE = 128
NUM_SEEDS   = 5
SEEDS       = list(range(NUM_SEEDS))
NUM_COMPARTMENTS = 4

# Lorenz-96 specific
L96_N    = 8
L96_F    = 8
L96_DT   = 0.05
SEQ_LEN  = 100
HORIZON  = 10
TRAJ_LEN = 50000

# Ablation variants to compare
# Each entry: (key, label)
ABLATION_VARIANTS = [
    ("no_ei",      "No EI Bistability (tanh)"),
    ("no_gate",    "No Shunting Gate (g≡1)"),
    ("no_wta",     "No WTA (α≡1/K)"),
]
VARIANT_KEYS = [v[0] for v in ABLATION_VARIANTS]

# ============================
# Lorenz-96 generator — RK4
# ============================

def lorenz96_rhs(state, F=L96_F):
    return ((np.roll(state, -1) - np.roll(state, 2))
            * np.roll(state, 1) - state + F)

def lorenz96_step(state, F=L96_F, dt=L96_DT):
    k1 = lorenz96_rhs(state,           F=F)
    k2 = lorenz96_rhs(state + dt/2*k1, F=F)
    k3 = lorenz96_rhs(state + dt/2*k2, F=F)
    k4 = lorenz96_rhs(state + dt*k3,   F=F)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def generate_lorenz96(T=TRAJ_LEN, N=L96_N, F=L96_F,
                      dt=L96_DT, warmup=2000):
    state    = F * np.ones(N)
    state[0] += 0.01

    for _ in range(warmup):
        state = lorenz96_step(state, F=F, dt=dt)
        if not np.isfinite(state).all():
            raise ValueError(
                f"L96 diverged during warmup. "
                f"Reduce dt (current={dt}) or F (current={F}).")

    traj = []
    for _ in range(T):
        state = lorenz96_step(state, F=F, dt=dt)
        traj.append(state.copy())

    traj = np.array(traj)
    if not np.isfinite(traj).all():
        raise ValueError("L96 trajectory contains NaN/Inf after warmup.")

    traj = (traj - traj.mean(axis=0)) / (traj.std(axis=0) + 1e-8)
    return traj

# ============================
# Dataset
# ============================

class Lorenz96Dataset(Dataset):
    def __init__(self, traj, indices, seq_len=SEQ_LEN,
                 prediction_horizon=HORIZON):
        self.data    = []
        self.targets = []
        for i in indices:
            x = traj[i:i + seq_len]
            y = traj[i + seq_len + prediction_horizon - 1]
            self.data.append(x)
            self.targets.append(y)
        self.data    = torch.tensor(np.array(self.data),    dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def make_train_val_datasets(seed, seq_len=SEQ_LEN,
                             prediction_horizon=HORIZON,
                             N=L96_N, F=L96_F,
                             n_train=1200, n_val=300):
    np.random.seed(seed)
    traj      = generate_lorenz96(T=TRAJ_LEN, N=N, F=F)
    max_start = len(traj) - seq_len - prediction_horizon

    stride     = max(1, seq_len // 4)
    all_starts = list(range(0, max_start, stride))
    split      = int(0.8 * len(all_starts))
    train_pool = all_starts[:split]

    last_train_end  = train_pool[-1] + seq_len + prediction_horizon
    first_val_start = last_train_end + seq_len
    val_pool = [s for s in all_starts[split:] if s >= first_val_start]

    if len(val_pool) < 50:
        raise ValueError(
            f"Only {len(val_pool)} val windows after gap. "
            f"Increase TRAJ_LEN or reduce SEQ_LEN.")

    np.random.shuffle(train_pool)
    np.random.shuffle(val_pool)
    train_idx = train_pool[:n_train]
    val_idx   = val_pool[:n_val]

    train_ds = Lorenz96Dataset(traj, train_idx, seq_len=seq_len,
                                prediction_horizon=prediction_horizon)
    val_ds   = Lorenz96Dataset(traj, val_idx,   seq_len=seq_len,
                                prediction_horizon=prediction_horizon)
    return train_ds, val_ds

# ============================
# DANU Ablation Cells
# ============================

class DANUCell_NoEI(nn.Module):
    """
    Ablation 1 — No EI bistability.
    Replaces the sigmoid (σ) + NMDA nonlinearity with a plain tanh.
    The compartment activation becomes:  d̃_k = tanh(u_k)
    All other components (gating, WTA attention) are unchanged.
    """
    def __init__(self, input_size, hidden_size, num_compartments):
        super().__init__()
        self.K = num_compartments
        self.H = hidden_size
        in_dim = input_size + hidden_size

        self.W_E      = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_I      = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_g      = nn.Linear(in_dim + hidden_size,
                                  hidden_size * num_compartments)
        self.lambda_I = nn.Parameter(
            torch.ones(num_compartments, hidden_size) * 0.1)
        self.ln       = nn.LayerNorm(hidden_size)
        # beta, theta, gamma removed — replaced by tanh
        self.W_q      = nn.Linear(hidden_size, hidden_size)
        self.W_out    = nn.Linear(hidden_size, hidden_size)
        self.ln_out   = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        B  = x.size(0)
        xh = torch.cat([x, h], dim=1)

        E = self.W_E(xh).view(B, self.K, self.H)
        I = self.W_I(xh).view(B, self.K, self.H)
        u = E - self.lambda_I * I
        u = self.ln(u)

        # ── Ablation: tanh instead of sigmoid+NMDA ──────────────────────────
        d_tilde = torch.tanh(u)                         # [B, K, H]

        d_bar   = d_tilde.mean(dim=1, keepdim=True)
        xh_dbar = torch.cat([xh, d_bar.squeeze(1)], dim=1)
        G       = torch.sigmoid(
            self.W_g(xh_dbar).view(B, self.K, self.H))
        d_gated = G * d_tilde

        q        = self.W_q(h).unsqueeze(2)
        scores   = torch.bmm(d_gated, q).squeeze(2) / (self.H**0.5)
        alpha    = torch.softmax(scores, dim=1)
        attended = (alpha.unsqueeze(2) * d_gated).sum(dim=1)

        h_new = torch.tanh(self.ln_out(self.W_out(attended) + h))
        return h_new


class DANUCell_NoGate(nn.Module):
    """
    Ablation 2 — No shunting gate.
    Sets G_k ≡ 1 for all compartments, so d_gated = d_tilde.
    W_g and associated parameters are removed entirely.
    EI bistability and WTA attention are unchanged.
    """
    def __init__(self, input_size, hidden_size, num_compartments):
        super().__init__()
        self.K = num_compartments
        self.H = hidden_size
        in_dim = input_size + hidden_size

        self.W_E      = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_I      = nn.Linear(in_dim, hidden_size * num_compartments)
        # W_g intentionally omitted
        self.lambda_I = nn.Parameter(
            torch.ones(num_compartments, hidden_size) * 0.1)
        self.ln       = nn.LayerNorm(hidden_size)
        self.beta     = nn.Parameter(torch.ones(num_compartments, 1))
        self.theta    = nn.Parameter(torch.zeros(num_compartments, 1))
        self.gamma    = nn.Parameter(
            torch.ones(num_compartments, 1) * 0.1)
        self.W_q      = nn.Linear(hidden_size, hidden_size)
        self.W_out    = nn.Linear(hidden_size, hidden_size)
        self.ln_out   = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        B  = x.size(0)
        xh = torch.cat([x, h], dim=1)

        E = self.W_E(xh).view(B, self.K, self.H)
        I = self.W_I(xh).view(B, self.K, self.H)
        u = E - self.lambda_I * I
        u = self.ln(u)

        sig     = torch.sigmoid(self.beta  * (u - self.theta))
        nmda    = self.gamma * (u**2) / (1.0 + u**2)
        d_tilde = sig + nmda

        # ── Ablation: skip shunting gate (G ≡ 1) ────────────────────────────
        d_gated = d_tilde                               # [B, K, H]

        q        = self.W_q(h).unsqueeze(2)
        scores   = torch.bmm(d_gated, q).squeeze(2) / (self.H**0.5)
        alpha    = torch.softmax(scores, dim=1)
        attended = (alpha.unsqueeze(2) * d_gated).sum(dim=1)

        h_new = torch.tanh(self.ln_out(self.W_out(attended) + h))
        return h_new


class DANUCell_NoWTA(nn.Module):
    """
    Ablation 3 — No WTA (winner-take-all) attention.
    Replaces softmax attention with uniform weights α_k = 1/K.
    W_q and associated parameters are removed entirely.
    EI bistability and shunting gate are unchanged.
    """
    def __init__(self, input_size, hidden_size, num_compartments):
        super().__init__()
        self.K = num_compartments
        self.H = hidden_size
        in_dim = input_size + hidden_size

        self.W_E      = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_I      = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_g      = nn.Linear(in_dim + hidden_size,
                                  hidden_size * num_compartments)
        self.lambda_I = nn.Parameter(
            torch.ones(num_compartments, hidden_size) * 0.1)
        self.ln       = nn.LayerNorm(hidden_size)
        self.beta     = nn.Parameter(torch.ones(num_compartments, 1))
        self.theta    = nn.Parameter(torch.zeros(num_compartments, 1))
        self.gamma    = nn.Parameter(
            torch.ones(num_compartments, 1) * 0.1)
        # W_q intentionally omitted
        self.W_out    = nn.Linear(hidden_size, hidden_size)
        self.ln_out   = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        B  = x.size(0)
        xh = torch.cat([x, h], dim=1)

        E = self.W_E(xh).view(B, self.K, self.H)
        I = self.W_I(xh).view(B, self.K, self.H)
        u = E - self.lambda_I * I
        u = self.ln(u)

        sig     = torch.sigmoid(self.beta  * (u - self.theta))
        nmda    = self.gamma * (u**2) / (1.0 + u**2)
        d_tilde = sig + nmda

        d_bar   = d_tilde.mean(dim=1, keepdim=True)
        xh_dbar = torch.cat([xh, d_bar.squeeze(1)], dim=1)
        G       = torch.sigmoid(
            self.W_g(xh_dbar).view(B, self.K, self.H))
        d_gated = G * d_tilde

        # ── Ablation: uniform weights instead of softmax ─────────────────────
        attended = d_gated.mean(dim=1)                  # [B, H]  (α = 1/K)

        h_new = torch.tanh(self.ln_out(self.W_out(attended) + h))
        return h_new


# ============================
# Generic DANU wrapper
# ============================

CELL_MAP = {
    "no_ei":   DANUCell_NoEI,
    "no_gate": DANUCell_NoGate,
    "no_wta":  DANUCell_NoWTA,
}

class DANUAblation(nn.Module):
    """Wraps any DANUCell variant with the standard RNN loop."""
    def __init__(self, input_size, hidden_size, output_size,
                 num_compartments, variant_key):
        super().__init__()
        self.hidden_size = hidden_size
        cell_cls  = CELL_MAP[variant_key]
        self.cell = cell_cls(input_size, hidden_size, num_compartments)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(h)

# ============================
# Training with early stopping
# ============================

def train_model(model, train_loader, val_loader,
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
                lr=5e-3, device="cpu",
                clip_grad=True, max_grad_norm=0.5):
    model.to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs, eta_min=lr / 50)
    loss_fn   = nn.MSELoss()
    hist      = {"train_loss": [], "val_loss": []}

    best_val         = float('inf')
    patience_counter = 0
    best_state       = None
    stopped_epoch    = max_epochs

    for epoch in range(max_epochs):
        model.train()
        tl, valid_batches = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)

            if not torch.isfinite(loss):
                print(f"  WARNING: NaN/Inf loss at epoch {epoch} — skipping batch")
                continue

            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            tl += loss.item()
            valid_batches += 1

        scheduler.step()

        if valid_batches == 0:
            print(f"  WARNING: entire epoch {epoch} had no valid batches")
            hist["train_loss"].append(float('nan'))
            hist["val_loss"].append(float('nan'))
            continue

        tl /= valid_batches

        model.eval()
        vl, valid_val_batches = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                if torch.isfinite(loss):
                    vl += loss.item()
                    valid_val_batches += 1

        if valid_val_batches == 0:
            hist["train_loss"].append(tl)
            hist["val_loss"].append(float('nan'))
            continue

        vl /= valid_val_batches
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)

        if vl < best_val:
            best_val         = vl
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                print(f"    Early stop at epoch {stopped_epoch} "
                      f"(best val={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist["stopped_epoch"] = stopped_epoch
    hist["best_val"]      = best_val
    return hist

# ============================
# RDA — Relative Degradation vs Ablation baseline
# ============================

def compute_rda(val_loader, model, device="cpu"):
    """
    Relative Degradation Analysis.

    For each sample compute squared error.  RDA is defined here as the
    normalised RMSE relative to a trivial zero-prediction baseline:

        RDA = RMSE_model / RMSE_baseline

    where RMSE_baseline = std of targets (i.e. predicting the mean).
    RDA < 1  → model beats the trivial baseline.
    RDA = 1  → model is no better than predicting the mean.

    We also return per-dimension MSE and the raw predictions for
    further analysis.
    """
    model.to(device)
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # [N, output_size]
    targets = np.concatenate(all_targets, axis=0)   # [N, output_size]

    # Per-sample squared error
    se          = (preds - targets) ** 2             # [N, output_size]
    mse_total   = se.mean()
    mse_per_dim = se.mean(axis=0)                    # [output_size]

    # Trivial baseline: predict the mean of each dimension
    baseline_se  = ((targets - targets.mean(axis=0)) ** 2).mean()
    rmse_model   = np.sqrt(mse_total)
    rmse_baseline = np.sqrt(baseline_se)
    rda           = rmse_model / (rmse_baseline + 1e-12)

    return {
        "mse":          float(mse_total),
        "mse_per_dim":  mse_per_dim.tolist(),
        "rmse":         float(rmse_model),
        "rmse_baseline": float(rmse_baseline),
        "rda":          float(rda),
    }

# ============================
# Multi-seed ablation experiment
# ============================

def run_ablation(dataset_name,
                 input_size, output_size,
                 hidden_size=HIDDEN_SIZE,
                 num_compartments=NUM_COMPARTMENTS,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 batch_size=BATCH_SIZE, device=DEVICE,
                 seeds=SEEDS,
                 ablation_variants=ABLATION_VARIANTS):

    variant_keys = [v[0] for v in ablation_variants]
    all_hists    = {k: [] for k in variant_keys}
    final_metrics = []

    print(f"\n{'='*80}")
    print(f"Running {dataset_name} — {len(seeds)} seeds")
    print(f"Ablation variants: {[v[1] for v in ablation_variants]}")
    print(f"K={num_compartments} compartments, hidden={hidden_size}")
    print(f"Early stopping: patience={patience}, max_epochs={max_epochs}")
    print(f"Checkpoints → {DRIVE_SAVE_DIR}")
    print(f"{'='*80}")

    for idx, seed in enumerate(seeds):
        print(f"\nSeed {idx+1}/{len(seeds)} (seed={seed})")

        cp = load_seed_checkpoint(idx, seed)
        if cp is not None:
            hists   = cp["model_hists"]
            metrics = cp["metrics"]
            for k in variant_keys:
                all_hists[k].append(hists[k])
            final_metrics.append(metrics)
            print(f"  Skipped — loaded from checkpoint.")
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset, val_dataset = make_train_val_datasets(
            seed=seed, seq_len=SEQ_LEN, prediction_horizon=HORIZON,
            N=L96_N, F=L96_F, n_train=1200, n_val=300)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                                  shuffle=False)

        seed_hists   = {}
        seed_metrics = {}

        for vkey, vlabel in ablation_variants:
            print(f"  Training {vlabel}...")
            model = DANUAblation(
                input_size, hidden_size, output_size,
                num_compartments=num_compartments,
                variant_key=vkey,
            ).to(device)
            hist = train_model(
                model, train_loader, val_loader,
                max_epochs=max_epochs, patience=patience,
                lr=5e-3, device=device,
                clip_grad=True, max_grad_norm=0.5)
            seed_hists[vkey] = hist

            # Compute RDA on the best-state model
            rda_metrics = compute_rda(val_loader, model, device=device)

            seed_metrics[f"{vkey}_best_val"]      = hist['best_val']
            seed_metrics[f"{vkey}_stopped_epoch"] = hist['stopped_epoch']
            seed_metrics[f"{vkey}_mse"]           = rda_metrics["mse"]
            seed_metrics[f"{vkey}_rda"]           = rda_metrics["rda"]
            seed_metrics[f"{vkey}_rmse"]          = rda_metrics["rmse"]
            seed_metrics[f"{vkey}_mse_per_dim"]   = rda_metrics["mse_per_dim"]

            print(f"    → stopped epoch {hist['stopped_epoch']}, "
                  f"best val MSE={hist['best_val']:.6f}, "
                  f"RDA={rda_metrics['rda']:.4f}")

        # Pairwise improvement between variants
        for vk_a, _ in ablation_variants:
            for vk_b, _ in ablation_variants:
                if vk_a != vk_b:
                    mse_a = seed_metrics[f"{vk_a}_mse"]
                    mse_b = seed_metrics[f"{vk_b}_mse"]
                    imp = ((mse_b - mse_a) / mse_b) * 100
                    seed_metrics[f"{vk_a}_vs_{vk_b}_improvement"] = imp

        save_seed_checkpoint(idx, seed, seed_hists, seed_metrics, dataset_name)

        for k in variant_keys:
            all_hists[k].append(seed_hists[k])
        final_metrics.append(seed_metrics)

        summary = " | ".join(
            [f"{vk}: MSE={seed_metrics[f'{vk}_mse']:.5f} "
             f"RDA={seed_metrics[f'{vk}_rda']:.3f}"
             for vk, _ in ablation_variants])
        print(f"  {summary}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def pad_and_stack(hist_list, key):
        arrays  = [np.array(h[key]) for h in hist_list]
        max_len = max(len(a) for a in arrays)
        padded  = [np.pad(a, (0, max_len - len(a)), mode='edge')
                   for a in arrays]
        return np.stack(padded, axis=0)

    results = {"final_metrics": final_metrics,
               "ablation_variants": ablation_variants}
    for k in variant_keys:
        hl = all_hists[k]
        results[f"{k}_train_mean"] = pad_and_stack(hl, 'train_loss').mean(axis=0)
        results[f"{k}_train_std"]  = pad_and_stack(hl, 'train_loss').std(axis=0)
        results[f"{k}_val_mean"]   = pad_and_stack(hl, 'val_loss').mean(axis=0)
        results[f"{k}_val_std"]    = pad_and_stack(hl, 'val_loss').std(axis=0)

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS — {dataset_name}")
    print(f"{'='*80}")
    print(f"\n{'Variant':<35} {'MSE mean±std':>20} {'RDA mean±std':>18} {'Epochs':>10}")
    print("-" * 85)
    for vkey, vlabel in ablation_variants:
        mses    = [m[f"{vkey}_mse"]           for m in final_metrics]
        rdas    = [m[f"{vkey}_rda"]           for m in final_metrics]
        epochs  = [m[f"{vkey}_stopped_epoch"] for m in final_metrics]
        print(f"  {vlabel:<33} "
              f"{np.mean(mses):.6f} ± {np.std(mses):.6f}   "
              f"{np.mean(rdas):.4f} ± {np.std(rdas):.4f}   "
              f"{np.mean(epochs):6.1f} ± {np.std(epochs):.1f}")

    def sig(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    stats_out = {}
    print(f"\nStatistical Significance — MSE (paired t-test):")
    for i, (vk_a, vl_a) in enumerate(ablation_variants):
        for vk_b, vl_b in ablation_variants[i+1:]:
            mses_a = [m[f"{vk_a}_mse"] for m in final_metrics]
            mses_b = [m[f"{vk_b}_mse"] for m in final_metrics]
            _, p   = stats.ttest_rel(mses_a, mses_b)
            key    = f"p_{vk_a}_vs_{vk_b}"
            stats_out[key] = p
            print(f"  {vl_a} vs {vl_b}: p = {p:.4f} {sig(p)}")

    print(f"\nStatistical Significance — RDA (paired t-test):")
    for i, (vk_a, vl_a) in enumerate(ablation_variants):
        for vk_b, vl_b in ablation_variants[i+1:]:
            rdas_a = [m[f"{vk_a}_rda"] for m in final_metrics]
            rdas_b = [m[f"{vk_b}_rda"] for m in final_metrics]
            _, p   = stats.ttest_rel(rdas_a, rdas_b)
            key    = f"p_rda_{vk_a}_vs_{vk_b}"
            stats_out[key] = p
            print(f"  {vl_a} vs {vl_b}: p = {p:.4f} {sig(p)}")

    results['statistics'] = stats_out
    save_final_results(results, dataset_name)
    return results

# ============================
# Plotting
# ============================

COLORS = {
    "no_ei":   '#A23B72',   # purple
    "no_gate": '#F18F01',   # amber
    "no_wta":  '#06A77D',   # teal
}

def visualize_results(results, title, save_path=None):
    ablation_variants = results["ablation_variants"]
    variant_keys      = [v[0] for v in ablation_variants]
    final_metrics     = results["final_metrics"]

    models = [(vk, vl, COLORS.get(vk, f"C{i}"))
              for i, (vk, vl) in enumerate(ablation_variants)]

    # ── Fig 1: learning curves ─────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig1.suptitle(title, fontsize=14, fontweight='bold', y=0.999)

    panel_cfg = [
        (axes[0, 0], "train", "log",    "Training Loss — log scale"),
        (axes[0, 1], "val",   "log",    "Validation Loss — log scale"),
        (axes[1, 0], "train", "linear", "Training Loss — linear scale"),
        (axes[1, 1], "val",   "linear", "Validation Loss — linear scale"),
    ]

    for ax, split, scale, panel_title in panel_cfg:
        for vk, vl, color in models:
            mean   = results[f"{vk}_{split}_mean"]
            std    = results[f"{vk}_{split}_std"]
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, label=vl, color=color, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.20, color=color)
        ax.set_title(panel_title, fontsize=12, fontweight='bold')
        ax.set_yscale(scale)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE Loss" + (" (log)" if scale == "log" else ""),
                      fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # ── Fig 2: MSE + RDA bar charts ────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(title + " — Ablation Summary", fontsize=13, fontweight='bold')

    x_pos  = np.arange(len(variant_keys))
    colors = [COLORS.get(vk, f"C{i}") for i, vk in enumerate(variant_keys)]
    labels = [vl for _, vl in ablation_variants]

    # MSE bars
    mse_means = [np.mean([m[f"{vk}_mse"] for m in final_metrics])
                 for vk in variant_keys]
    mse_stds  = [np.std( [m[f"{vk}_mse"] for m in final_metrics])
                 for vk in variant_keys]
    axes2[0].bar(x_pos, mse_means, yerr=mse_stds, capsize=6,
                 color=colors, alpha=0.80)
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels(labels, fontsize=9, rotation=10, ha='right')
    axes2[0].set_ylabel('MSE (val)', fontsize=11)
    axes2[0].set_title('Validation MSE', fontsize=12, fontweight='bold')
    axes2[0].grid(alpha=0.3, axis='y')

    # Significance brackets on MSE
    if 'statistics' in results:
        y_max = max(m + s for m, s in zip(mse_means, mse_stds)) * 1.05
        pairs = [(i, j)
                 for i in range(len(variant_keys))
                 for j in range(i+1, len(variant_keys))]
        for lvl, (i, j) in enumerate(pairs):
            vk_a = variant_keys[i]
            vk_b = variant_keys[j]
            key  = f"p_{vk_a}_vs_{vk_b}"
            if key not in results['statistics']:
                continue
            p     = results['statistics'][key]
            stars = ('***' if p < 0.001 else '**' if p < 0.01
                     else '*' if p < 0.05 else 'n.s.')
            y = y_max * (1.00 + lvl * 0.12)
            axes2[0].plot([i, j], [y, y], 'k-', linewidth=1)
            axes2[0].text((i + j) / 2, y * 1.01, stars,
                          ha='center', fontsize=11, fontweight='bold')

    # RDA bars
    rda_means = [np.mean([m[f"{vk}_rda"] for m in final_metrics])
                 for vk in variant_keys]
    rda_stds  = [np.std( [m[f"{vk}_rda"] for m in final_metrics])
                 for vk in variant_keys]
    axes2[1].bar(x_pos, rda_means, yerr=rda_stds, capsize=6,
                 color=colors, alpha=0.80)
    axes2[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.6,
                     label='Trivial baseline (RDA=1)')
    axes2[1].set_xticks(x_pos)
    axes2[1].set_xticklabels(labels, fontsize=9, rotation=10, ha='right')
    axes2[1].set_ylabel('RDA  (RMSE / RMSE_baseline)', fontsize=11)
    axes2[1].set_title('Relative Degradation (RDA)', fontsize=12,
                        fontweight='bold')
    axes2[1].legend(fontsize=9)
    axes2[1].grid(alpha=0.3, axis='y')

    # Per-dimension MSE heatmap
    n_dims  = L96_N
    heatmap = np.zeros((len(variant_keys), n_dims))
    for i, vk in enumerate(variant_keys):
        per_dim_list = [m[f"{vk}_mse_per_dim"] for m in final_metrics]
        heatmap[i]   = np.mean(per_dim_list, axis=0)

    im = axes2[2].imshow(heatmap, aspect='auto', cmap='viridis')
    axes2[2].set_xticks(np.arange(n_dims))
    axes2[2].set_xticklabels([f"dim {i}" for i in range(n_dims)],
                               fontsize=8, rotation=45, ha='right')
    axes2[2].set_yticks(np.arange(len(variant_keys)))
    axes2[2].set_yticklabels(labels, fontsize=9)
    axes2[2].set_title('Per-Dimension MSE (mean over seeds)',
                        fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes2[2], label='MSE')

    fig2.tight_layout()

    if save_path:
        base         = save_path.replace('.png', '')
        curves_path  = base + '_curves.png'
        bars_path    = base + '_bars.png'
        drive_curves = drive_path(os.path.basename(curves_path))
        drive_bars   = drive_path(os.path.basename(bars_path))
        for fig, lpath, dpath in [
            (fig1, curves_path, drive_curves),
            (fig2, bars_path,   drive_bars),
        ]:
            fig.savefig(lpath, dpi=300, bbox_inches='tight')
            fig.savefig(dpath, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved locally:  {curves_path}  |  {bars_path}")
        print(f"Plots saved to Drive: {drive_curves}  |  {drive_bars}")

    plt.show()

# ============================
# Reload helper
# ============================

def reload_all_checkpoints(dataset_name, seeds=SEEDS,
                            ablation_variants=ABLATION_VARIANTS):
    variant_keys  = [v[0] for v in ablation_variants]
    all_hists     = {k: [] for k in variant_keys}
    final_metrics = []

    for idx, seed in enumerate(seeds):
        cp = load_seed_checkpoint(idx, seed)
        if cp is None:
            raise FileNotFoundError(
                f"No checkpoint for seed_idx={idx}, seed={seed}")
        for k in variant_keys:
            all_hists[k].append(cp["model_hists"][k])
        final_metrics.append(cp["metrics"])

    def pad_and_stack(hist_list, key):
        arrays  = [np.array(h[key]) for h in hist_list]
        max_len = max(len(a) for a in arrays)
        padded  = [np.pad(a, (0, max_len - len(a)), mode='edge')
                   for a in arrays]
        return np.stack(padded, axis=0)

    results = {"final_metrics": final_metrics,
               "ablation_variants": ablation_variants}
    for k in variant_keys:
        hl = all_hists[k]
        results[f"{k}_train_mean"] = pad_and_stack(hl, 'train_loss').mean(axis=0)
        results[f"{k}_train_std"]  = pad_and_stack(hl, 'train_loss').std(axis=0)
        results[f"{k}_val_mean"]   = pad_and_stack(hl, 'val_loss').mean(axis=0)
        results[f"{k}_val_std"]    = pad_and_stack(hl, 'val_loss').std(axis=0)

    print(f"Reloaded {len(seeds)} checkpoints from {DRIVE_SAVE_DIR}")
    return results

# ============================
# Main
# ============================

if __name__ == "__main__":

    # from google.colab import drive
    # drive.mount('/content/drive')

    print("="*80)
    print("DANU Ablation Study — Lorenz-96")
    print(f"K={NUM_COMPARTMENTS} compartments | {NUM_SEEDS} seeds")
    print(f"Variants: {[v[1] for v in ABLATION_VARIANTS]}")
    print(f"L96: N={L96_N}, F={L96_F}, dt={L96_DT}  (RK4)")
    print(f"Task: seq_len={SEQ_LEN}, horizon={HORIZON}, hidden={HIDDEN_SIZE}")
    print(f"Saving to: {DRIVE_SAVE_DIR}")
    print("="*80)

    DATASET_NAME = (f"Lorenz-96 N={L96_N} F={L96_F} "
                    f"— seq={SEQ_LEN}, h={HORIZON} "
                    f"[DANU K={NUM_COMPARTMENTS} ablation]")

    results = run_ablation(
        DATASET_NAME,
        input_size=L96_N,
        output_size=L96_N,
        hidden_size=HIDDEN_SIZE,
        num_compartments=NUM_COMPARTMENTS,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        seeds=SEEDS,
        ablation_variants=ABLATION_VARIANTS,
    )

    visualize_results(
        results,
        f"DANU Ablation (K={NUM_COMPARTMENTS}, Lorenz-96 N={L96_N} F={L96_F}): "
        f"MSE & RDA — {NUM_SEEDS} seeds, early stop, RK4",
        save_path=drive_path("danu_ablation_results.png"),
    )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print(f"All results saved to: {DRIVE_SAVE_DIR}")
    print("="*80)