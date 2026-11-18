


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
# ===========================a=
DRIVE_SAVE_DIR = "lorenz96_danu_compartments5"
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
NUM_SEEDS   = 10
SEEDS       = list(range(NUM_SEEDS))

# Lorenz-96 specific
L96_N    = 8
L96_F    = 8
L96_DT   = 0.05
SEQ_LEN  = 100
HORIZON  = 10
TRAJ_LEN = 50000

# Compartment configs being compared
COMPARTMENT_CONFIGS = [4] #[2, 3]   # DANU-2 vs DANU-3

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

    train_ds = Lorenz96Dataset(traj, train_idx,
                                seq_len=seq_len,
                                prediction_horizon=prediction_horizon)
    val_ds   = Lorenz96Dataset(traj, val_idx,
                                seq_len=seq_len,
                                prediction_horizon=prediction_horizon)
    return train_ds, val_ds

# ============================
# DANU Model (parameterised compartments)
# ============================

class DANUCell(nn.Module):
    """
    Dendritic Attentive Neural Unit cell.
    num_compartments is fully configurable — supports 2, 3, 4, or any K.
    hidden_size must be divisible by num_compartments only for the
    TractableDendritic variant; DANU does NOT require that constraint
    because compartment outputs are H-dim and aggregated via attention,
    not concatenated.
    """
    def __init__(self, input_size, hidden_size, num_compartments):
        super().__init__()
        self.K = num_compartments
        self.H = hidden_size
        in_dim = input_size + hidden_size

        self.W_E = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_I = nn.Linear(in_dim, hidden_size * num_compartments)
        self.W_g = nn.Linear(in_dim + hidden_size,
                             hidden_size * num_compartments)
        self.lambda_I = nn.Parameter(
            torch.ones(num_compartments, hidden_size) * 0.1)
        self.ln    = nn.LayerNorm(hidden_size)
        self.beta  = nn.Parameter(torch.ones(num_compartments, 1))
        self.theta = nn.Parameter(torch.zeros(num_compartments, 1))
        self.gamma = nn.Parameter(
            torch.ones(num_compartments, 1) * 0.1)
        self.W_q    = nn.Linear(hidden_size, hidden_size)
        self.W_out  = nn.Linear(hidden_size, hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        B  = x.size(0)
        xh = torch.cat([x, h], dim=1)

        E = self.W_E(xh).view(B, self.K, self.H)
        I = self.W_I(xh).view(B, self.K, self.H)
        u = E - self.lambda_I * I
        u = self.ln(u)

        sig  = torch.sigmoid(self.beta  * (u - self.theta))
        nmda = self.gamma * (u**2) / (1.0 + u**2)
        d_tilde = sig + nmda

        d_bar   = d_tilde.mean(dim=1, keepdim=True)
        xh_dbar = torch.cat([xh, d_bar.squeeze(1)], dim=1)
        G       = torch.sigmoid(
            self.W_g(xh_dbar).view(B, self.K, self.H))
        d_gated = G * d_tilde

        q      = self.W_q(h).unsqueeze(2)
        scores = torch.bmm(d_gated, q).squeeze(2) / (self.H**0.5)
        alpha  = torch.softmax(scores, dim=1)
        attended = (alpha.unsqueeze(2) * d_gated).sum(dim=1)

        h_new = torch.tanh(self.ln_out(self.W_out(attended) + h))
        return h_new


class DANU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_compartments):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DANUCell(input_size, hidden_size, num_compartments)
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
        tl            = 0.0
        valid_batches = 0

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
        vl                = 0.0
        valid_val_batches = 0
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
# Multi-seed experiment
# ============================

def run_comparison(dataset_name,
                   input_size, output_size,
                   hidden_size=HIDDEN_SIZE,
                   max_epochs=MAX_EPOCHS, patience=PATIENCE,
                   batch_size=BATCH_SIZE, device=DEVICE,
                   seeds=SEEDS,
                   compartment_configs=COMPARTMENT_CONFIGS):

    # Dict keyed by number of compartments
    all_hists     = {k: [] for k in compartment_configs}
    final_metrics = []

    print(f"\n{'='*80}")
    print(f"Running {dataset_name} — {len(seeds)} seeds")
    print(f"Comparing DANU compartments: {compartment_configs}")
    print(f"input_size={input_size}, output_size={output_size}, hidden={hidden_size}")
    print(f"Early stopping: patience={patience}, max_epochs={max_epochs}")
    print(f"Checkpoints → {DRIVE_SAVE_DIR}")
    print(f"{'='*80}")

    for idx, seed in enumerate(seeds):
        print(f"\nSeed {idx+1}/{len(seeds)} (seed={seed})")

        cp = load_seed_checkpoint(idx, seed)
        if cp is not None:
            hists   = cp["model_hists"]
            metrics = cp["metrics"]
            for k in compartment_configs:
                all_hists[k].append(hists[f"danu_{k}"])
            final_metrics.append(metrics)
            print(f"  Skipped — loaded from checkpoint.")
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset, val_dataset = make_train_val_datasets(
            seed=seed, seq_len=SEQ_LEN, prediction_horizon=HORIZON,
            N=L96_N, F=L96_F, n_train=1200, n_val=300)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        seed_hists   = {}
        seed_bests   = {}

        for K in compartment_configs:
            print(f"  Training DANU-{K} ({K} compartments)...")
            model = DANU(input_size, hidden_size,
                         output_size=output_size,
                         num_compartments=K).to(device)
            hist = train_model(
                model, train_loader, val_loader,
                max_epochs=max_epochs, patience=patience,
                lr=5e-3, device=device,
                clip_grad=True, max_grad_norm=0.5)
            seed_hists[f"danu_{K}"] = hist
            seed_bests[K]           = hist['best_val']
            print(f"    → stopped epoch {hist['stopped_epoch']}, "
                  f"best val={hist['best_val']:.6f}")

        # Per-seed metrics
        seed_metrics = {}
        for K in compartment_configs:
            seed_metrics[f"danu_{K}_best"]          = seed_bests[K]
            seed_metrics[f"danu_{K}_stopped_epoch"] = seed_hists[f"danu_{K}"]['stopped_epoch']

        # Pairwise improvement: each K vs every other K
        for i, K_a in enumerate(compartment_configs):
            for K_b in compartment_configs:
                if K_a != K_b:
                    imp = ((seed_bests[K_b] - seed_bests[K_a]) / seed_bests[K_b]) * 100
                    seed_metrics[f"danu_{K_a}_vs_{K_b}_improvement"] = imp

        save_seed_checkpoint(idx, seed, seed_hists, seed_metrics, dataset_name)

        for K in compartment_configs:
            all_hists[K].append(seed_hists[f"danu_{K}"])
        final_metrics.append(seed_metrics)

        summary = " | ".join([f"DANU-{K}: {seed_bests[K]:.6f}"
                               for K in compartment_configs])
        print(f"  {summary}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def pad_and_stack(hist_list, key):
        arrays  = [np.array(h[key]) for h in hist_list]
        max_len = max(len(a) for a in arrays)
        padded  = [np.pad(a, (0, max_len - len(a)), mode='edge')
                   for a in arrays]
        return np.stack(padded, axis=0)

    results = {"final_metrics": final_metrics}
    for K in compartment_configs:
        hl = all_hists[K]
        results[f"danu{K}_train_mean"] = pad_and_stack(hl, 'train_loss').mean(axis=0)
        results[f"danu{K}_train_std"]  = pad_and_stack(hl, 'train_loss').std(axis=0)
        results[f"danu{K}_val_mean"]   = pad_and_stack(hl, 'val_loss').mean(axis=0)
        results[f"danu{K}_val_std"]    = pad_and_stack(hl, 'val_loss').std(axis=0)

    bests = {K: [m[f"danu_{K}_best"] for m in final_metrics]
             for K in compartment_configs}

    # All paired t-tests
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS — {dataset_name}")
    print(f"{'='*80}")

    print(f"\nBest Validation Loss (mean ± std):")
    for K in compartment_configs:
        print(f"  DANU-{K}: {np.mean(bests[K]):.6f} ± {np.std(bests[K]):.6f}")

    print(f"\nEpochs to convergence (mean ± std):")
    for K in compartment_configs:
        vals = [m[f"danu_{K}_stopped_epoch"] for m in final_metrics]
        print(f"  DANU-{K}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

    def sig(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    stats_out = {}
    print(f"\nStatistical Significance (paired t-test):")
    for i, K_a in enumerate(compartment_configs):
        for K_b in compartment_configs[i+1:]:
            _, p = stats.ttest_rel(bests[K_a], bests[K_b])
            key  = f"p_danu{K_a}_vs_danu{K_b}"
            stats_out[key] = p
            print(f"  DANU-{K_a} vs DANU-{K_b}: p = {p:.4f} {sig(p)}")

    print(f"\nRelative improvement (positive = row better than column):")
    for K_a in compartment_configs:
        for K_b in compartment_configs:
            if K_a != K_b:
                imps = [m[f"danu_{K_a}_vs_{K_b}_improvement"]
                        for m in final_metrics]
                print(f"  DANU-{K_a} vs DANU-{K_b}: "
                      f"{np.mean(imps):+.2f}% ± {np.std(imps):.2f}%")

    results['statistics'] = stats_out
    save_final_results(results, dataset_name)
    return results

# ============================
# Plotting
# ============================

COLORS = {
    2: '#1B4332',   # dark green  — DANU-2
    3: '#F18F01',   # amber       — DANU-3
    4: '#06A77D',   # teal        — DANU-4 (if ever added)
}

def visualize_results(results, title, compartment_configs=COMPARTMENT_CONFIGS,
                      save_path=None):

    models = [(K, f"DANU-{K}", COLORS.get(K, f"C{i}"))
              for i, K in enumerate(compartment_configs)]

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
        for K, label, color in models:
            mean = results[f"danu{K}_{split}_mean"]
            std  = results[f"danu{K}_{split}_std"]
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, label=label, color=color, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.20, color=color)
        ax.set_title(panel_title, fontsize=12, fontweight='bold')
        ax.set_yscale(scale)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE Loss" + (" (log)" if scale == "log" else ""),
                      fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # ── Fig 2: final performance ───────────────────────────────────────────
    final_metrics = results['final_metrics']
    bests = {K: [m[f"danu_{K}_best"] for m in final_metrics]
             for K in compartment_configs}

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle(title + " — Final Performance", fontsize=13, fontweight='bold')

    # Bar chart
    x_pos  = np.arange(len(compartment_configs))
    means  = [np.mean(bests[K]) for K in compartment_configs]
    stds   = [np.std(bests[K])  for K in compartment_configs]
    colors = [COLORS.get(K, f"C{i}") for i, K in enumerate(compartment_configs)]

    axes2[0].bar(x_pos, means, yerr=stds, capsize=6,
                 color=colors, alpha=0.80)
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels([f"DANU-{K}" for K in compartment_configs],
                              fontsize=12)
    axes2[0].set_ylabel('Best Validation Loss (MSE)', fontsize=11)
    axes2[0].set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    axes2[0].grid(alpha=0.3, axis='y')

    # Significance brackets
    if 'statistics' in results:
        y_max = max(m + s for m, s in zip(means, stds)) * 1.05
        pairs = [(i, j)
                 for i in range(len(compartment_configs))
                 for j in range(i+1, len(compartment_configs))]
        for level_idx, (i, j) in enumerate(pairs):
            K_a  = compartment_configs[i]
            K_b  = compartment_configs[j]
            key  = f"p_danu{K_a}_vs_danu{K_b}"
            if key not in results['statistics']:
                continue
            p    = results['statistics'][key]
            stars = ('***' if p < 0.001 else '**' if p < 0.01
                     else '*' if p < 0.05 else 'n.s.')
            y = y_max * (1.00 + level_idx * 0.12)
            axes2[0].plot([i, j], [y, y], 'k-', linewidth=1)
            axes2[0].text((i + j) / 2, y * 1.01, stars,
                          ha='center', fontsize=11, fontweight='bold')

    # Improvement bars
    imp_labels, imp_means, imp_stds, imp_colors = [], [], [], []
    for K_a in compartment_configs:
        for K_b in compartment_configs:
            if K_a != K_b:
                key  = f"danu_{K_a}_vs_{K_b}_improvement"
                vals = [m[key] for m in final_metrics]
                imp_labels.append(f"DANU-{K_a}\nvs DANU-{K_b}")
                imp_means.append(np.mean(vals))
                imp_stds.append(np.std(vals))
                imp_colors.append(COLORS.get(K_a, 'grey'))

    x_imp = np.arange(len(imp_labels))
    axes2[1].bar(x_imp, imp_means, yerr=imp_stds, capsize=6,
                 color=imp_colors, alpha=0.80)
    axes2[1].set_xticks(x_imp)
    axes2[1].set_xticklabels(imp_labels, fontsize=10)
    axes2[1].set_ylabel('Improvement (%)', fontsize=11)
    axes2[1].set_title('Pairwise Improvement (mean ± std)', fontsize=12, fontweight='bold')
    axes2[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes2[1].grid(alpha=0.3, axis='y')

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
                            compartment_configs=COMPARTMENT_CONFIGS):
    all_hists     = {K: [] for K in compartment_configs}
    final_metrics = []

    for idx, seed in enumerate(seeds):
        cp = load_seed_checkpoint(idx, seed)
        if cp is None:
            raise FileNotFoundError(
                f"No checkpoint for seed_idx={idx}, seed={seed}")
        for K in compartment_configs:
            all_hists[K].append(cp["model_hists"][f"danu_{K}"])
        final_metrics.append(cp["metrics"])

    def pad_and_stack(hist_list, key):
        arrays  = [np.array(h[key]) for h in hist_list]
        max_len = max(len(a) for a in arrays)
        padded  = [np.pad(a, (0, max_len - len(a)), mode='edge')
                   for a in arrays]
        return np.stack(padded, axis=0)

    results = {"final_metrics": final_metrics}
    for K in compartment_configs:
        hl = all_hists[K]
        results[f"danu{K}_train_mean"] = pad_and_stack(hl, 'train_loss').mean(axis=0)
        results[f"danu{K}_train_std"]  = pad_and_stack(hl, 'train_loss').std(axis=0)
        results[f"danu{K}_val_mean"]   = pad_and_stack(hl, 'val_loss').mean(axis=0)
        results[f"danu{K}_val_std"]    = pad_and_stack(hl, 'val_loss').std(axis=0)

    print(f"Reloaded {len(seeds)} checkpoints from {DRIVE_SAVE_DIR}")
    return results

# ============================
# Main
# ============================

if __name__ == "__main__":

    # from google.colab import drive
    # drive.mount('/content/drive')

    print("="*80)
    print("DANU Compartment Comparison: 2 vs 3  —  Lorenz-96")
    print(f"L96:  N={L96_N}, F={L96_F}, dt={L96_DT}  (RK4 integration)")
    print(f"Task: seq_len={SEQ_LEN}, horizon={HORIZON}, hidden={HIDDEN_SIZE}")
    print(f"Compartments tested: {COMPARTMENT_CONFIGS}")
    print(f"Saving to: {DRIVE_SAVE_DIR}")
    print("="*80)

    DATASET_NAME = (f"Lorenz-96 N={L96_N} F={L96_F} "
                    f"— seq={SEQ_LEN}, h={HORIZON} ")
                    #f"[DANU {COMPARTMENT_CONFIGS[0]} vs {COMPARTMENT_CONFIGS[1]}]")

    results = run_comparison(
        DATASET_NAME,
        input_size=L96_N,
        output_size=L96_N,
        hidden_size=HIDDEN_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        seeds=SEEDS,
        compartment_configs=COMPARTMENT_CONFIGS,
    )

    visualize_results(
        results,
        f"Lorenz-96 (N={L96_N}, F={L96_F}): DANU-2 vs DANU-3 "
        f"— seq={SEQ_LEN}, h={HORIZON} (10 seeds, early stop, RK4)",
        compartment_configs=COMPARTMENT_CONFIGS,
        save_path=drive_path("danu_compartments_comparison.png"),
    )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print(f"All results saved to: {DRIVE_SAVE_DIR}")
    print("="*80)
