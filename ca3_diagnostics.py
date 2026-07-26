"""
Several diagnostic to test that the simulations do not saturate
"""

import math
import numpy as np


def _dominance_wins(spk_recorder, asm_ids, asm_sizes, n_asm, bin_ms, window_start, window_end, target_idx):
    """Like the private _dominance_frac helper inside run_cued_dominance_probe,
    but returns raw (wins, n_bins) instead of a fraction. Kept as a module-level
    function (rather than nested) so both the single- and multi-seed probes can
    call it without duplicating the binning logic.
    """
    import nest

    ev = nest.GetStatus(spk_recorder, "events")[0]
    senders = np.asarray(ev["senders"])
    times = np.asarray(ev["times"])
    edges = np.arange(window_start, window_end + bin_ms, bin_ms)
    n_bins = len(edges) - 1
    if n_bins <= 0:
        return 0, 0
    wins = 0
    for bi in range(n_bins):
        t0, t1 = edges[bi], edges[bi + 1]
        mask = (times >= t0) & (times < t1)
        s = senders[mask]
        rates = [
            np.isin(s, asm_ids[ai]).sum() / (asm_sizes[ai] * (bin_ms / 1000.0))
            for ai in range(n_asm)
        ]
        if int(np.argmax(rates)) == target_idx:
            wins += 1
    return wins, n_bins


def run_cued_dominance_probe(
    build_fn,
    cue_assembly_idx=0,
    cue_frac=0.3,
    cue_rate=800.0,
    cue_start=200.0,
    cue_dur=50.0,
    sim_ms=800.0,
    bin_ms=50.0,
    post_window_ms=300.0,
    seed_connect=42,
    build_kwargs=None,
):
    """MODIFIED: same behavior as the original, plus exact win/bin counts
    in the return dict (wins_cued, n_bins_cued, wins_baseline, n_bins_baseline)
    so run_cued_dominance_multi_seed_probe can pool without rounding.
    See original docstring in ca3_diagnostics.py for the full rationale
    (fast winner-take-all switching washes out plain rate averages).
    """
    import nest

    build_kwargs = dict(build_kwargs or {})
    build_kwargs.setdefault("seed_connect", seed_connect)

    # --- cued run ---
    net = build_fn(**build_kwargs)
    n_asm = len(net["ca3_assemblies"])
    asm_ids = [np.array(a, dtype=int) for a in net["ca3_assemblies"]]
    asm_sizes = [len(a) for a in asm_ids]

    cue_asm = asm_ids[cue_assembly_idx]
    rng = np.random.default_rng(seed_connect + 999)
    n_cue = max(1, int(len(cue_asm) * cue_frac))
    cued_ids = rng.choice(cue_asm, size=n_cue, replace=False)

    cue_gen = nest.Create(
        "poisson_generator", len(cued_ids),
        params={"rate": float(cue_rate), "start": float(cue_start), "stop": float(cue_start + cue_dur)},
    )
    n_cue_conns = len(cued_ids)
    nest.Connect(
        cue_gen.tolist(), cued_ids.tolist(), conn_spec="one_to_one",
        syn_spec={"weight": np.full(n_cue_conns, 3.0), "delay": np.full(n_cue_conns, 1.0)},
    )
    nest.Simulate(float(sim_ms))

    post_start = cue_start + cue_dur
    post_end = post_start + post_window_ms
    wins_cued, n_bins_cued = _dominance_wins(
        net["spk_ca3_pyr"], asm_ids, asm_sizes, n_asm, bin_ms, post_start, post_end, cue_assembly_idx
    )
    dominance_frac_cued = (wins_cued / n_bins_cued) if n_bins_cued else float("nan")

    # --- matched uncued baseline run (same seed -> identical wiring, no cue) ---
    net_base = build_fn(**build_kwargs)
    nest.Simulate(float(sim_ms))
    wins_baseline, n_bins_baseline = _dominance_wins(
        net_base["spk_ca3_pyr"], asm_ids, asm_sizes, n_asm, bin_ms, 0.0, float(sim_ms), cue_assembly_idx
    )
    baseline_dominance_frac = (wins_baseline / n_bins_baseline) if n_bins_baseline else float("nan")

    return {
        "dominance_frac_cued": dominance_frac_cued,
        "baseline_dominance_frac": baseline_dominance_frac,
        "null_uniform_frac": 1.0 / n_asm,
        "difference": dominance_frac_cued - baseline_dominance_frac,
        # new: exact counts for pooling
        "wins_cued": wins_cued,
        "n_bins_cued": n_bins_cued,
        "wins_baseline": wins_baseline,
        "n_bins_baseline": n_bins_baseline,
    }


def run_cued_dominance_multi_seed_probe(
    build_fn,
    seeds,
    cue_assembly_idx=0,
    cue_frac=0.3,
    cue_rate=800.0,
    cue_start=200.0,
    cue_dur=50.0,
    sim_ms=800.0,
    bin_ms=50.0,
    post_window_ms=300.0,
    build_kwargs=None,
):
    """Pool the dominance/win-count test across many independently-wired
    networks (different seed_connect -> different assembly membership and
    connectivity draw each time), instead of trusting one trial's handful
    of post-cue bins.

    Runs run_cued_dominance_probe once per seed, sums wins_cued/n_bins_cued
    and wins_baseline/n_bins_baseline as exact integers (no rounding), then
    tests the pooled cued fraction against (a) the uniform null 1/n_assemblies
    and (b) the pooled matched-baseline fraction, via a normal approximation
    to the binomial (z-test, p-value from math.erf -- no scipy needed).

    Returns per_seed detail plus pooled_frac_cued, pooled_frac_baseline,
    null_uniform_frac, z_cued_vs_null, p_cued_vs_null, z_cued_vs_baseline,
    p_cued_vs_baseline.
    """
    per_seed = []
    total_wins_cued = 0
    total_bins_cued = 0
    total_wins_baseline = 0
    total_bins_baseline = 0
    null_p = None

    for seed in seeds:
        kwargs = dict(build_kwargs or {})
        result = run_cued_dominance_probe(
            build_fn=build_fn,
            cue_assembly_idx=cue_assembly_idx,
            cue_frac=cue_frac,
            cue_rate=cue_rate,
            cue_start=cue_start,
            cue_dur=cue_dur,
            sim_ms=sim_ms,
            bin_ms=bin_ms,
            post_window_ms=post_window_ms,
            seed_connect=seed,
            build_kwargs=kwargs,
        )
        if null_p is None:
            null_p = result["null_uniform_frac"]

        total_wins_cued += result["wins_cued"]
        total_bins_cued += result["n_bins_cued"]
        total_wins_baseline += result["wins_baseline"]
        total_bins_baseline += result["n_bins_baseline"]

        per_seed.append({
            "seed": seed,
            "dominance_frac_cued": result["dominance_frac_cued"],
            "baseline_dominance_frac": result["baseline_dominance_frac"],
            "wins_cued": result["wins_cued"],
            "n_bins_cued": result["n_bins_cued"],
            "wins_baseline": result["wins_baseline"],
            "n_bins_baseline": result["n_bins_baseline"],
        })

    pooled_frac_cued = (total_wins_cued / total_bins_cued) if total_bins_cued else float("nan")
    pooled_frac_baseline = (
        total_wins_baseline / total_bins_baseline if total_bins_baseline else float("nan")
    )

    def _z_to_p(z):
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    # one-sample: pooled cued fraction vs uniform null
    if total_bins_cued and null_p is not None:
        se_null = math.sqrt(null_p * (1 - null_p) / total_bins_cued)
        z_cued_vs_null = (pooled_frac_cued - null_p) / se_null if se_null else float("nan")
        p_cued_vs_null = _z_to_p(z_cued_vs_null)
    else:
        z_cued_vs_null = float("nan")
        p_cued_vs_null = float("nan")

    # two-sample pooled-proportion test: cued vs matched baseline
    n1, n2 = total_bins_cued, total_bins_baseline
    if n1 and n2:
        p_pool = (total_wins_cued + total_wins_baseline) / (n1 + n2)
        se_2samp = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
        z_cued_vs_baseline = (pooled_frac_cued - pooled_frac_baseline) / se_2samp if se_2samp else float("nan")
        p_cued_vs_baseline = _z_to_p(z_cued_vs_baseline)
    else:
        z_cued_vs_baseline = float("nan")
        p_cued_vs_baseline = float("nan")

    return {
        "seeds": list(seeds),
        "per_seed": per_seed,
        "pooled_frac_cued": pooled_frac_cued,
        "pooled_frac_baseline": pooled_frac_baseline,
        "null_uniform_frac": null_p,
        "total_bins_cued": total_bins_cued,
        "total_bins_baseline": total_bins_baseline,
        "z_cued_vs_null": z_cued_vs_null,
        "p_cued_vs_null": p_cued_vs_null,
        "z_cued_vs_baseline": z_cued_vs_baseline,
        "p_cued_vs_baseline": p_cued_vs_baseline,
    }
    
"""
Patch for ca3_diagnostics.py -- cue phase-sweep probe
-------------------------------------------------------
Follows directly from run_cued_dominance_multi_seed_probe's pooled null
result at fixed cue_start=200.0 (pooled_frac_cued=0.208 vs null=0.25,
p=0.29 vs null, p=0.47 vs baseline, n=20 seeds).

That result rules out "not enough seeds" as the explanation for the
earlier single-trial noise, but it doesn't yet rule in or out the
timing hypothesis: with a ~100-150ms winner-take-all period and no
per-seed phase alignment, cue_start=200.0 lands at whatever phase each
seed's own cycle happens to be in -- effectively random across seeds.
A real phase-locked completion effect (cue helps if it lands during
the "decision window" between bursts, does nothing or actively
interferes if it lands mid-burst) would average out to exactly the
flat null already measured.

This holds the cued neurons and everything else fixed and sweeps only
cue_start across one assumed cycle period, pooling the SAME seed set
at each phase offset with the existing dominance/win-count machinery.
An efficiency fix vs. naively calling run_cued_dominance_multi_seed_probe
once per phase: the uncued baseline for a given seed doesn't depend on
cue_start at all, so it's computed once per seed and reused across every
phase offset, instead of being recomputed n_phases times for no
statistical benefit.

Requires _dominance_wins from the earlier dominance-probe patch to
already be in ca3_diagnostics.py.
"""

import math
import numpy as np


def run_cue_phase_sweep_probe(
    build_fn,
    seeds,
    cue_assembly_idx=0,
    cue_frac=0.3,
    cue_rate=800.0,
    cue_dur=50.0,
    sim_ms=800.0,
    bin_ms=50.0,
    post_window_ms=300.0,
    cycle_period_ms=125.0,
    n_phases=5,
    phase_start_ms=150.0,
    build_kwargs=None,
):
    """Sweep cue_start across one assumed switching-cycle period
    (cycle_period_ms, default 125ms -- the midpoint of the ~100-150ms
    range read off the earlier per-bin argmax/argmin table) in n_phases
    even steps starting at phase_start_ms, pooling the same `seeds` at
    each phase offset.

    cycle_period_ms is a guess, not a measured quantity -- this sweep is
    itself partly how you'd refine it. If one phase offset shows a real
    peak, a finer sweep centered there (smaller phase steps, narrower
    range) is the natural follow-up before trusting an exact optimum.

    Multiple comparisons: with n_phases independent significance tests,
    treat any single "best" phase's p-value as suggestive, not
    confirmatory -- the returned bonferroni_alpha field is the threshold
    a phase's p_cued_vs_null would need to clear to hold up under a
    simple Bonferroni correction for n_phases comparisons. A promising
    phase should be re-tested against a fresh, non-overlapping seed set
    before treating it as a confirmed result.

    Cost: 1 baseline + n_phases cued sims per seed (vs. 2 * n_phases if
    you naively called run_cued_dominance_multi_seed_probe once per
    phase, which would needlessly re-simulate an identical baseline at
    every phase for the same seed).

    Returns a dict with phase_offsets_ms, cue_starts, per-phase pooled
    results (pooled_frac_cued, pooled_frac_baseline, z_cued_vs_null,
    p_cued_vs_null, z_cued_vs_baseline, p_cued_vs_baseline), plus
    best_phase_idx and bonferroni_alpha for convenience.
    """
    import nest

    phase_offsets = [i * (cycle_period_ms / n_phases) for i in range(n_phases)]
    cue_starts = [phase_start_ms + off for off in phase_offsets]

    wins_cued_by_phase = [0] * n_phases
    bins_cued_by_phase = [0] * n_phases
    total_wins_baseline = 0
    total_bins_baseline = 0
    null_p = None
    per_seed_detail = []

    for seed in seeds:
        kwargs = dict(build_kwargs or {})
        kwargs.setdefault("seed_connect", seed)

        # --- baseline: computed ONCE per seed, reused across all phases ---
        net_base = build_fn(**kwargs)
        n_asm = len(net_base["ca3_assemblies"])
        if null_p is None:
            null_p = 1.0 / n_asm
        asm_ids_base = [np.array(a, dtype=int) for a in net_base["ca3_assemblies"]]
        asm_sizes_base = [len(a) for a in asm_ids_base]
        nest.Simulate(float(sim_ms))
        w_base, n_base = _dominance_wins(
            net_base["spk_ca3_pyr"], asm_ids_base, asm_sizes_base, n_asm,
            bin_ms, 0.0, float(sim_ms), cue_assembly_idx,
        )
        total_wins_baseline += w_base
        total_bins_baseline += n_base

        seed_phase_wins = []
        for pi, cue_start in enumerate(cue_starts):
            net = build_fn(**kwargs)
            asm_ids = [np.array(a, dtype=int) for a in net["ca3_assemblies"]]
            asm_sizes = [len(a) for a in asm_ids]
            cue_asm = asm_ids[cue_assembly_idx]
            rng = np.random.default_rng(seed + 999)
            n_cue = max(1, int(len(cue_asm) * cue_frac))
            cued_ids = rng.choice(cue_asm, size=n_cue, replace=False)

            cue_gen = nest.Create(
                "poisson_generator", len(cued_ids),
                params={"rate": float(cue_rate), "start": float(cue_start), "stop": float(cue_start + cue_dur)},
            )
            n_cue_conns = len(cued_ids)
            nest.Connect(
                cue_gen.tolist(), cued_ids.tolist(), conn_spec="one_to_one",
                syn_spec={"weight": np.full(n_cue_conns, 3.0), "delay": np.full(n_cue_conns, 1.0)},
            )
            nest.Simulate(float(sim_ms))

            post_start = cue_start + cue_dur
            post_end = post_start + post_window_ms
            w, n = _dominance_wins(
                net["spk_ca3_pyr"], asm_ids, asm_sizes, n_asm, bin_ms,
                post_start, post_end, cue_assembly_idx,
            )
            wins_cued_by_phase[pi] += w
            bins_cued_by_phase[pi] += n
            seed_phase_wins.append({"cue_start": cue_start, "wins": w, "n_bins": n})

        per_seed_detail.append({
            "seed": seed,
            "baseline_wins": w_base,
            "baseline_bins": n_base,
            "phase_wins": seed_phase_wins,
        })

    pooled_frac_baseline = (
        total_wins_baseline / total_bins_baseline if total_bins_baseline else float("nan")
    )

    def _z_to_p(z):
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    phase_results = []
    for pi in range(n_phases):
        n1 = bins_cued_by_phase[pi]
        pooled_frac_cued = (wins_cued_by_phase[pi] / n1) if n1 else float("nan")

        if n1 and null_p is not None:
            se_null = math.sqrt(null_p * (1 - null_p) / n1)
            z_null = (pooled_frac_cued - null_p) / se_null if se_null else float("nan")
            p_null = _z_to_p(z_null)
        else:
            z_null = float("nan")
            p_null = float("nan")

        n2 = total_bins_baseline
        if n1 and n2:
            p_pool = (wins_cued_by_phase[pi] + total_wins_baseline) / (n1 + n2)
            se_2s = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
            z_base = (pooled_frac_cued - pooled_frac_baseline) / se_2s if se_2s else float("nan")
            p_base = _z_to_p(z_base)
        else:
            z_base = float("nan")
            p_base = float("nan")

        phase_results.append({
            "phase_offset_ms": phase_offsets[pi],
            "cue_start": cue_starts[pi],
            "pooled_frac_cued": pooled_frac_cued,
            "wins_cued": wins_cued_by_phase[pi],
            "n_bins_cued": n1,
            "pooled_frac_baseline": pooled_frac_baseline,
            "z_cued_vs_null": z_null,
            "p_cued_vs_null": p_null,
            "z_cued_vs_baseline": z_base,
            "p_cued_vs_baseline": p_base,
        })

    p_values = [r["p_cued_vs_null"] for r in phase_results]
    finite_idxs = [i for i, p in enumerate(p_values) if not math.isnan(p)]
    best_phase_idx = min(finite_idxs, key=lambda i: p_values[i]) if finite_idxs else None

    return {
        "seeds": list(seeds),
        "cycle_period_ms": cycle_period_ms,
        "phase_offsets_ms": phase_offsets,
        "cue_starts": cue_starts,
        "phase_results": phase_results,
        "per_seed_detail": per_seed_detail,
        "null_uniform_frac": null_p,
        "best_phase_idx": best_phase_idx,
        "bonferroni_alpha": (0.05 / n_phases) if n_phases else float("nan"),
    }
