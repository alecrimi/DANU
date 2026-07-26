#!/usr/bin/env python3
"""
CA1-like microcircuit in NEST using *only* Izhikevich neurons + spike generators as inputs.

This is a modified version of the code of Dr. Max Talanov: https://github.com/max-talanov/tinyHippo

Key points (per NEST izhikevich model):
- Incoming spikes change V_m directly by the synaptic weight (no PSC kernel inside this neuron model).
- So weights are in mV "jumps" and can be much smaller than iaf_psc_exp currents.

Populations:
  PYR    : Regular Spiking (RS) Izhikevich params
  BASKET : Fast Spiking (FS) params
  OLM    : Low-Threshold Spiking-ish (LTS) params (proxy for O-LM style inhibition)

Inputs (generators only):
  CA3-like independent Poisson -> PYR and BASKET
  ECIII-like independent Poisson -> PYR
"""

import nest
import numpy as np


# -------------------------
# Helpers
# -------------------------

def safe_set_seeds(master_seed: int = 20260111):
    n_threads = nest.GetKernelStatus().get("local_num_threads", 1)
    try:
        nest.SetKernelStatus(
            {
                "grng_seed": master_seed,
                "rng_seeds": list(range(master_seed + 1, master_seed + 1 + n_threads)),
            }
        )
    except Exception:
        # older fallback
        try:
            nest.SetKernelStatus({"rng_seed": master_seed})
        except Exception:
            pass


def bernoulli_connect(pre, post, p, weight, delay, rng, extra_delays=None, extra_weight_scale=1.0):
    """Explicit Bernoulli connectivity; easy to hack for motif studies.

    extra_delays: optional iterable of additional delays (ms). When given,
    every connected pair also gets one extra synapse per delay in this
    list, at `weight * extra_weight_scale`, in addition to the primary
    synapse at `delay`. This is the "duplicated synapse" trick used to
    approximate a slow (NMDA-like) depolarizing tail on top of the
    izhikevich model's instantaneous V_m-jump synapses -- see
    structured_ee_connect for the fuller rationale.
    """
    pre_ids = np.array(pre, dtype=int)
    post_ids = np.array(post, dtype=int)
    for src in pre_ids:
        mask = rng.random(post_ids.size) < p
        targets = post_ids[mask].tolist()
        if not targets:
            continue
        nest.Connect([int(src)], targets, syn_spec={"weight": float(weight), "delay": float(delay)})
        if extra_delays:
            w_extra = float(weight) * float(extra_weight_scale)
            for d_extra in extra_delays:
                nest.Connect([int(src)], targets, syn_spec={"weight": w_extra, "delay": float(d_extra)})


def assign_ca3_assemblies(CA3_PYR, n_assemblies, overlap_frac, seed):
    """Partition CA3 pyramidal cells into n_assemblies groups.

    This is the structural piece a flat Bernoulli(p) recurrent graph is
    missing: without it, there's no way for the network to hold multiple,
    separable stored patterns -- only one global mode (silence / noise /
    one synchronized burst). Real CA3 "memories" correspond to specific
    ensembles of cells with elevated mutual connectivity; this creates that.

    overlap_frac > 0 lets assemblies share some members (pattern
    interference / partial overlap studies); 0 gives disjoint assemblies.
    """
    ids = np.array(CA3_PYR.tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    base_groups = np.array_split(ids, int(n_assemblies))

    if overlap_frac <= 0:
        return [np.array(g) for g in base_groups]

    assemblies = []
    for i, g in enumerate(base_groups):
        extra = []
        for j, g2 in enumerate(base_groups):
            if j == i:
                continue
            n_extra = int(len(g2) * overlap_frac)
            if n_extra > 0:
                extra.extend(rng.choice(g2, size=n_extra, replace=False))
        assemblies.append(np.unique(np.concatenate([g, extra])))
    return assemblies


def structured_ee_connect(
    ca3_ids, id_to_assembly, p_within, p_between, weight, delay, rng,
    syn_model="static_synapse", syn_params=None,
    extra_delays=None, extra_weight_scale=1.0,
):
    """CA3 E->E connectivity with two probabilities instead of one: dense
    WITHIN an assembly (the recurrent mass that lets a partial cue complete
    a whole pattern) and sparse BETWEEN assemblies (keeps patterns from
    blurring into each other). Self-connections are excluded.

    extra_delays: optional iterable of additional delays (ms), each adding
    one more synapse per connected pair at `weight * extra_weight_scale`.
    Without this, every recurrent E->E spike in the izhikevich model is a
    single instantaneous V_m jump with no decay -- there is nothing to
    "remember" a spike beyond the postsynaptic neuron's own membrane time
    constant. Staggering copies of the same connection across several
    delays (e.g. +5/+15/+30 ms) turns one presynaptic spike into a longer
    depolarizing tail, approximating the slow NMDA-like component real CA3
    recurrent excitation leans on for sustained/attractor-like activity.
    """
    ca3_ids = np.array(ca3_ids, dtype=int)
    labels = np.array([id_to_assembly[int(i)] for i in ca3_ids])

    def _spec(w, d):
        s = {"weight": float(w), "delay": float(d)}
        if syn_model != "static_synapse":
            s["synapse_model"] = syn_model
            if syn_params:
                s.update(syn_params)
        return s

    for idx, src in enumerate(ca3_ids):
        same_mask = labels == labels[idx]
        probs = np.where(same_mask, p_within, p_between)
        draws = rng.random(len(ca3_ids))
        mask = (draws < probs) & (ca3_ids != src)
        targets = ca3_ids[mask].tolist()
        if not targets:
            continue
        nest.Connect([int(src)], targets, syn_spec=_spec(weight, delay))
        if extra_delays:
            w_extra = float(weight) * float(extra_weight_scale)
            for d_extra in extra_delays:
                nest.Connect([int(src)], targets, syn_spec=_spec(w_extra, d_extra))


def conn_stats(label: str, pre, post):
    conns = nest.GetConnections(pre, post)
    n_pre, n_post = len(pre), len(post)
    try:
        n_conn = conns.get("source").size
    except Exception:
        n_conn = len(conns)
    density = n_conn / (n_pre * n_post) if n_pre * n_post > 0 else 0.0
    print(
        f"{label:12s}: {n_conn:7d} connections | density={density:.4f} | "
        f"avg outdegree={n_conn/n_pre:.2f} | avg indegree={n_conn/n_post:.2f}"
    )


def mean_rate(pop, spk, sim_ms: float) -> float:
    ev = nest.GetStatus(spk, "events")[0]
    return len(ev["senders"]) / (len(pop) * (sim_ms / 1000.0))



# -------------------------
# Theta modulation helper
# -------------------------

def maybe_make_theta_generators(n: int, rate_mean: float, rate_amp: float, freq_hz: float, phase: float = 0.0):
    """Create N sinusoidally-modulated Poisson generators (if available in this NEST build).

    Returns:
        NodeCollection of generators, or None if the model is unavailable.
    """
    try:
        gens = nest.Create(
            "sinusoidal_poisson_generator",
            int(n),
            params={
                "rate": float(rate_mean),
                "amplitude": float(rate_amp),
                "frequency": float(freq_hz),
                "phase": float(phase),
            },
        )
        return gens
    except Exception as e:
        print(f"[theta] sinusoidal_poisson_generator not available ({e}); running without theta.")
        return None
# ------------
def make_swr_event_generators(
    n: int,
    start_ms: float,
    stop_ms: float,
    sharpwave_rate: float,
    ripple_rate_mean: float,
    ripple_rate_amp: float,
    ripple_hz: float,
    ripple_phase: float = 0.0,
):
    """Create generators for a single SWR event.

    We model SWR as:
      1) a 'sharp wave' (slow depolarizing) drive: Poisson at sharpwave_rate between start/stop
      2) a 'ripple' drive: ideally sinusoidally-modulated Poisson at ripple_hz between start/stop
         (falls back to plain Poisson if sinusoidal_poisson_generator is unavailable).

    Returns:
        (sharp_gen, ripple_gen) NodeCollections, each of size n (or ripple_gen may be None).
    """
    sharp = nest.Create(
        "poisson_generator",
        int(n),
        params={"rate": float(sharpwave_rate), "start": float(start_ms), "stop": float(stop_ms)},
    )

    ripple = None
    try:
        ripple = nest.Create(
            "sinusoidal_poisson_generator",
            int(n),
            params={
                "rate": float(ripple_rate_mean),
                "amplitude": float(ripple_rate_amp),
                "frequency": float(ripple_hz),
                "phase": float(ripple_phase),
                "start": float(start_ms),
                "stop": float(stop_ms),
            },
        )
    except Exception as e:
        # Fallback: constant high-rate Poisson during the ripple window
        print(f"[swr] sinusoidal_poisson_generator not available ({e}); using poisson_generator for ripple.")
        ripple = nest.Create(
            "poisson_generator",
            int(n),
            params={"rate": float(ripple_rate_mean), "start": float(start_ms), "stop": float(stop_ms)},
        )
    return sharp, ripple
# -----------
# Build CA1 microcircuit
# -------------------------

def build_ca1_izh(
    N_pyr=200,
    N_basket=15,
    N_olm=12,
    p_EE=0.02,
    p_EI=0.10,
    p_IE=0.15,
    p_OE=0.10,
    rate_ca3_pyr=1200.0,   # Hz per generator (independent per neuron)
    rate_ec_pyr=900.0,
    rate_ca3_ba=1200.0,
    local_num_threads=4,
):
    nest.ResetKernel()
    nest.SetKernelStatus(
        {
            "resolution": 0.1,        # ms
            "local_num_threads": int(local_num_threads),
            "print_time": True,
            "overwrite_files": True,
        }
    )
    safe_set_seeds()

    if "izhikevich" not in nest.Models("nodes"):
        raise RuntimeError("NEST model 'izhikevich' not found in this installation.")

    ''' --- Izhikevich parameter sets (canonical-ish)'''
    # RS (regular spiking)
    pyr_params = dict(a=0.02, b=0.2, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    # FS (fast spiking interneuron proxy)
    basket_params = dict(a=0.1, b=0.2, c=-65.0, d=2.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    # LTS-ish (slow / adapting interneuron proxy; used here as OLM-like)
    olm_params = dict(a=0.02, b=0.25, c=-65.0, d=2.0, V_m=-65.0, U_m=-16.25, I_e=0.0)

    PYR = nest.Create("izhikevich", N_pyr, params=pyr_params)
    BASKET = nest.Create("izhikevich", N_basket, params=basket_params)
    OLM = nest.Create("izhikevich", N_olm, params=olm_params)

    ''' --- Independent external inputs (generators only) '''
    CA3_to_PYR = nest.Create("poisson_generator", N_pyr, params={"rate": float(rate_ca3_pyr)})
    ECIII_to_PYR = nest.Create("poisson_generator", N_pyr, params={"rate": float(rate_ec_pyr)})
    CA3_to_BA = nest.Create("poisson_generator", N_basket, params={"rate": float(rate_ca3_ba)})

    # --- Synapse weights (IMPORTANT: for izhikevich, weight directly jumps V_m in mV)
    w_ca3_pyr = 3.0
    w_ec_pyr = 2.0
    w_ca3_basket = 3.0

    w_pyr_pyr = 0.8
    w_pyr_basket = 1.5
    w_pyr_olm = 1.2

    w_basket_pyr = -5.0
    w_basket_basket = -4.0
    w_olm_pyr = -3.0

    d_fast = 1.5
    d_slow = 3.0

    # --- Connect inputs one-to-one
    nest.Connect(CA3_to_PYR, PYR, conn_spec="one_to_one", syn_spec={"weight": w_ca3_pyr, "delay": d_fast})
    nest.Connect(ECIII_to_PYR, PYR, conn_spec="one_to_one", syn_spec={"weight": w_ec_pyr, "delay": d_slow})
    nest.Connect(CA3_to_BA, BASKET, conn_spec="one_to_one", syn_spec={"weight": w_ca3_basket, "delay": d_fast})

    # --- Recurrent connectivity
    rng = np.random.default_rng(42)

    bernoulli_connect(PYR, PYR, p_EE, w_pyr_pyr, d_fast, rng)           # PYR->PYR
    bernoulli_connect(PYR, BASKET, p_EI, w_pyr_basket, d_fast, rng)     # PYR->Basket
    bernoulli_connect(PYR, OLM, 0.08, w_pyr_olm, d_slow, rng)           # PYR->OLM

    bernoulli_connect(BASKET, PYR, p_IE, w_basket_pyr, d_fast, rng)     # Basket->PYR
    bernoulli_connect(BASKET, BASKET, 0.10, w_basket_basket, d_fast, rng)

    bernoulli_connect(OLM, PYR, p_OE, w_olm_pyr, d_slow, rng)           # OLM->PYR

    # --- Recorders
    spk_pyr = nest.Create("spike_recorder")
    spk_ba = nest.Create("spike_recorder")
    spk_olm = nest.Create("spike_recorder")
    nest.Connect(PYR, spk_pyr)
    nest.Connect(BASKET, spk_ba)
    nest.Connect(OLM, spk_olm)

    # Vm traces (plot per neuron to avoid "diagonal ramps" artifact)
    try:
        vm = nest.Create("multimeter", params={"record_from": ["V_m", "U_m"], "interval": 0.2})
    except Exception:
        vm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.2})
    pyr_probe = PYR[:5]
    nest.Connect(vm, pyr_probe)

    # Connectivity stats
    print("\nConnectivity stats:")
    conn_stats("PYR->PYR", PYR, PYR)
    conn_stats("PYR->BA", PYR, BASKET)
    conn_stats("PYR->OLM", PYR, OLM)
    conn_stats("BA->PYR", BASKET, PYR)
    conn_stats("OLM->PYR", OLM, PYR)

    return dict(
        PYR=PYR,
        BASKET=BASKET,
        OLM=OLM,
        spk_pyr=spk_pyr,
        spk_ba=spk_ba,
        spk_olm=spk_olm,
        vm=vm,
        pyr_probe=pyr_probe,
    )


# -------------------------
# Build CA1 + CA3 microcircuit (CA3 recurrent + Schaffer collaterals)
# -------------------------

def build_ca1_ca3_izh(
    # --- CA1 sizes
    N_ca1_pyr=200,
    N_ca1_basket=15,
    N_ca1_olm=12,
    # --- CA3 sizes
    N_ca3_pyr=300,
    N_ca3_int=41,
    # --- CA1 recurrent probabilities
    p_ca1_EE=0.02,
    p_ca1_EI=0.10,
    p_ca1_IE=0.15,
    p_ca1_OE=0.10,
    # --- CA3 recurrent probabilities (stronger recurrence than CA1)
    p_ca3_EE=0.04,
    p_ca3_EI=0.12,
    p_ca3_IE=0.20,
    p_ca3_II=0.10,
    # --- CA3 -> CA1 (Schaffer collateral) probabilities
    p_schaffer_pyr=0.10,
    p_schaffer_basket=0.12,
    # --- External drive rates (Hz, independent poisson per neuron)
    rate_ec_ca1_pyr=900.0,
    # NOTE on the three CA3 PYR feedforward rates below: EC/DG/generic-drive
    # are each independent per-neuron Poisson generators -- identical
    # statistics for every CA3 PYR neuron regardless of assembly. At the
    # original defaults (600/800/1000 Hz) this tonic, assembly-blind drive
    # appears to dominate CA3E firing rate almost entirely (see
    # ca3_diagnostics.run_recurrence_ablation_probe): quadrupling effective
    # within-assembly recurrent weight via delay-copies barely moved CV
    # (0.044 -> 0.050) or the cued-recall ratio (1.01 -> 0.98), which is the
    # signature of feedforward-dominated firing, not insufficient temporal
    # smearing. rate_dg_ca3_pyr and rate_ca3_drive_pyr are cut here as a
    # first attempt to shift that balance -- they're the two purely-tonic
    # "keep the network active" drives, as opposed to rate_ec_ca3_pyr which
    # is left as the nominal cue/pattern-bearing input. This is a hypothesis,
    # not a verified fix -- rerun run_recurrence_ablation_probe and the
    # attractor tests after changing these before trusting them.
    rate_ca3_drive_pyr=150.0,
    rate_ec_ca3_pyr=800.0,
    rate_dg_ca3_pyr=300.0,
    rate_drive_ca1_basket=1200.0,
    rate_drive_ca3_int=1200.0,
    # --- Theta rhythm (optional, via sinusoidally-modulated Poisson drive)
    theta_on: bool = False,
    theta_hz: float = 8.0,
    # mean/amplitude rates (Hz) per neuron
    theta_rate_mean_ca3: float = 1200.0,
    theta_rate_amp_ca3: float = 1100.0,
    theta_rate_mean_ca1: float = 1200.0,
    theta_rate_amp_ca1: float = 1100.0,
    # weights (mV jump per theta spike)
    theta_w_ca3_pyr: float = 0.80,
    theta_w_ca3_int: float = 1.80,
    theta_w_ca1_pyr: float = 1.00,
    theta_w_ca1_int: float = 2.00,
    theta_delay: float = 1.0,
    # --- SWR events (optional; toy approximation of sharp-wave ripple bursts)
    swr_on: bool = False,
    # list of (start_ms, stop_ms) tuples; if None and swr_on=True, a default event is used
    swr_events=None,
    # generator parameters
    swr_sharpwave_rate: float = 300.0,
    swr_ripple_hz: float = 180.0,
    swr_ripple_rate_mean: float = 1200.0,
    swr_ripple_rate_amp: float = 900.0,
    # weights (mV jump per SWR spike)
    swr_w_sharp_ca3_pyr: float = 0.35,
    swr_w_ripple_ca3_int: float = 0.60,
    swr_w_ripple_ca3_pyr: float = 0.15,
    swr_w_sharp_ca1_pyr: float = 0.20,
    swr_w_ripple_ca1_basket: float = 0.55,
    swr_w_ripple_ca1_pyr: float = 0.10,
    # --- CA3 E->E structured (assembly) connectivity
    ca3_ee_structured: bool = True,
    n_ca3_assemblies: int = 4,
    ca3_assembly_overlap: float = 0.0,
    p_ca3_EE_within: float = 0.55,
    p_ca3_EE_between: float = 0.015,
    # --- CA3 E->E short-term facilitation (tsodyks_synapse)
    ca3_ee_facilitating: bool = True,
    ca3_ee_tsodyks_params=None,
    # --- CA3 recurrent E->E weight (exposed for drive-vs-recurrence sweeps;
    # was hardcoded to 0.9 previously)
    # Raised alongside the feedforward-rate cuts above, for the same reason:
    # at 0.9 mV/spike, recurrent excitation was too small a fraction of
    # total input to differentiate assemblies even when the feedforward
    # rates dominating it are reduced. Still a hypothesis pending
    # run_recurrence_ablation_probe / the attractor tests -- not verified.
    w_ca3_EE: float = 2.5,
    # --- CA3 pyramidal spike-frequency adaptation (Izhikevich 'd') and
    # reset voltage ('c'), exposed separately from CA1 since high adaptation
    # + a deeply negative reset actively fight sustained up-state firing.
    # Previously these were hardcoded to CA1's RS values (c=-65, d=8) with
    # only 'd' nominally exposed but left at the RS default -- i.e. CA3 and
    # CA1 pyramidal cells had identical dynamics no matter what was passed.
    # Defaults now sit closer to Izhikevich's own "intrinsically bursting"
    # (IB) set (c=-55, d=4) rather than plain RS, since some capacity for
    # sustained/bursting firing is a prerequisite for attractor-like
    # persistence -- RS's strong adaptation (d=8) actively suppresses it.
    ca3_pyr_c: float = -55.0,
    ca3_pyr_d: float = 2.0,
    # --- Number of delayed copies used to approximate a slow (NMDA-like)
    # component on CA3 E->E synapses. The 'izhikevich' model has no PSC
    # kernel -- spikes jump V_m instantaneously with no decay -- so a single
    # recurrent spike carries no temporal "memory" beyond the postsynaptic
    # neuron's own membrane dynamics (~10-20ms). Splitting each E->E
    # connection into N copies at staggered delays smears a presynaptic
    # spike into a longer depolarizing tail, which is what lets recurrent
    # excitation actually accumulate rather than acting as a series of
    # isolated instantaneous kicks. n=1 reproduces the old single-delay
    # behavior; default is now 4 (primary + all three staggered extras)
    # since at n=1 this mechanism was previously wired up but never
    # actually invoked -- see structured_ee_connect / bernoulli_connect.
    ca3_ee_n_delay_copies: int = 4,
    ca3_ee_extra_delays_ms=(5.0, 15.0, 30.0),
    ca3_ee_extra_delay_weight_scale: float = 0.5,
    # --- RNG seed
    seed_connect=42,
    # --- Kernel threads (set to 1 when running many sims in parallel processes)
    local_num_threads=4,
):
    """A compact CA1+CA3 toy network in the same style as build_ca1_izh.

    CA3 is modeled as a recurrent excitatory (pyramidal) + inhibitory (basket-like) network.
    CA3 exc projects to CA1 PYR and CA1 BASKET via Schaffer-collateral-like connections.

    Notes:
      - Uses NEST 'izhikevich' only; synaptic weights are direct V_m jumps in mV.
      - This is *not* a full CA3 microanatomy (mossy fibers, multiple interneuron classes, etc.).
        It's the minimal recurrent E/I core you can later refine.
      - CA3 E->E is now assembly-structured by default (ca3_ee_structured=True): dense
        within-assembly connectivity (p_ca3_EE_within) + sparse between-assembly
        connectivity (p_ca3_EE_between), instead of flat Bernoulli(p_ca3_EE). Set
        ca3_ee_structured=False to fall back to the old flat behavior using p_ca3_EE.
    """

    nest.ResetKernel()
    nest.SetKernelStatus(
        {
            "resolution": 0.1,
            "local_num_threads": int(local_num_threads),
            "print_time": True,
            "overwrite_files": True,
        }
    )
    safe_set_seeds()

    if "izhikevich" not in nest.Models("nodes"):
        raise RuntimeError("NEST model 'izhikevich' not found in this installation.")

    # --- Izhikevich parameter sets
    pyr_params = dict(a=0.02, b=0.2, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    ca3_pyr_params = dict(
        a=0.02, b=0.2, c=float(ca3_pyr_c), d=float(ca3_pyr_d), V_m=-65.0, U_m=-13.0, I_e=0.0
    )
    basket_params = dict(a=0.1, b=0.2, c=-65.0, d=2.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    olm_params = dict(a=0.02, b=0.25, c=-65.0, d=2.0, V_m=-65.0, U_m=-16.25, I_e=0.0)

    # --- Populations
    CA1_PYR = nest.Create("izhikevich", int(N_ca1_pyr), params=pyr_params)
    CA1_BASKET = nest.Create("izhikevich", int(N_ca1_basket), params=basket_params)
    CA1_OLM = nest.Create("izhikevich", int(N_ca1_olm), params=olm_params)

    CA3_PYR = nest.Create("izhikevich", int(N_ca3_pyr), params=ca3_pyr_params)
    CA3_INT = nest.Create("izhikevich", int(N_ca3_int), params=basket_params)

    # --- External inputs (independent poisson per neuron)
    ECIII_to_CA1 = nest.Create("poisson_generator", len(CA1_PYR), params={"rate": float(rate_ec_ca1_pyr)})
    DRIVE_to_CA1_BA = nest.Create(
        "poisson_generator", len(CA1_BASKET), params={"rate": float(rate_drive_ca1_basket)}
    )

    EC_to_CA3 = nest.Create("poisson_generator", len(CA3_PYR), params={"rate": float(rate_ec_ca3_pyr)})
    DG_to_CA3 = nest.Create("poisson_generator", len(CA3_PYR), params={"rate": float(rate_dg_ca3_pyr)})
    DRIVE_to_CA3 = nest.Create("poisson_generator", len(CA3_PYR), params={"rate": float(rate_ca3_drive_pyr)})
    DRIVE_to_CA3_INT = nest.Create(
        "poisson_generator", len(CA3_INT), params={"rate": float(rate_drive_ca3_int)}
    )

    # --- Synapse weights (mV jumps)
    w_ec = 2.0
    w_dg = 3.0
    w_drive = 2.0

    w_ca3_EI = 1.6
    w_ca3_IE = -5.5
    w_ca3_II = -4.5

    w_ca1_EE = 0.8
    w_ca1_EI = 1.5
    w_ca1_EO = 1.2
    w_ca1_IE = -5.0
    w_ca1_II = -4.0
    w_ca1_OE = -3.0

    w_schaffer_pyr = 2.4
    w_schaffer_ba = 2.8

    d_fast = 1.5
    d_slow = 3.0

    nest.Connect(ECIII_to_CA1, CA1_PYR, conn_spec="one_to_one", syn_spec={"weight": w_ec, "delay": d_slow})
    nest.Connect(
        DRIVE_to_CA1_BA, CA1_BASKET, conn_spec="one_to_one", syn_spec={"weight": w_drive, "delay": d_fast}
    )

    nest.Connect(EC_to_CA3, CA3_PYR, conn_spec="one_to_one", syn_spec={"weight": w_ec, "delay": d_slow})
    nest.Connect(DG_to_CA3, CA3_PYR, conn_spec="one_to_one", syn_spec={"weight": w_dg, "delay": d_fast})
    nest.Connect(DRIVE_to_CA3, CA3_PYR, conn_spec="one_to_one", syn_spec={"weight": w_drive, "delay": d_fast})
    nest.Connect(
        DRIVE_to_CA3_INT, CA3_INT, conn_spec="one_to_one", syn_spec={"weight": w_drive, "delay": d_fast}
    )

    if theta_on:
        th_ca3_pyr = maybe_make_theta_generators(len(CA3_PYR), theta_rate_mean_ca3, theta_rate_amp_ca3, theta_hz)
        if th_ca3_pyr is not None:
            nest.Connect(
                th_ca3_pyr, CA3_PYR, conn_spec="one_to_one",
                syn_spec={"weight": float(theta_w_ca3_pyr), "delay": float(theta_delay)},
            )
        th_ca3_int = maybe_make_theta_generators(len(CA3_INT), theta_rate_mean_ca3, theta_rate_amp_ca3, theta_hz)
        if th_ca3_int is not None:
            nest.Connect(
                th_ca3_int, CA3_INT, conn_spec="one_to_one",
                syn_spec={"weight": float(theta_w_ca3_int), "delay": float(theta_delay)},
            )

        th_ca1_pyr = maybe_make_theta_generators(len(CA1_PYR), theta_rate_mean_ca1, theta_rate_amp_ca1, theta_hz)
        if th_ca1_pyr is not None:
            nest.Connect(
                th_ca1_pyr, CA1_PYR, conn_spec="one_to_one",
                syn_spec={"weight": float(theta_w_ca1_pyr), "delay": float(theta_delay)},
            )
        th_ca1_ba = maybe_make_theta_generators(len(CA1_BASKET), theta_rate_mean_ca1, theta_rate_amp_ca1, theta_hz)
        if th_ca1_ba is not None:
            nest.Connect(
                th_ca1_ba, CA1_BASKET, conn_spec="one_to_one",
                syn_spec={"weight": float(theta_w_ca1_int), "delay": float(theta_delay)},
            )
        th_ca1_olm = maybe_make_theta_generators(len(CA1_OLM), theta_rate_mean_ca1, theta_rate_amp_ca1, theta_hz)
        if th_ca1_olm is not None:
            nest.Connect(
                th_ca1_olm, CA1_OLM, conn_spec="one_to_one",
                syn_spec={"weight": float(theta_w_ca1_int), "delay": float(theta_delay)},
            )

    if swr_on:
        if swr_events is None:
            swr_events = [(550.0, 630.0)]

        for (swr_start, swr_stop) in swr_events:
            sw_sharp_ca3, sw_ripple_ca3 = make_swr_event_generators(
                n=len(CA3_PYR),
                start_ms=swr_start,
                stop_ms=swr_stop,
                sharpwave_rate=swr_sharpwave_rate,
                ripple_rate_mean=swr_ripple_rate_mean,
                ripple_rate_amp=swr_ripple_rate_amp,
                ripple_hz=swr_ripple_hz,
            )
            nest.Connect(
                sw_sharp_ca3, CA3_PYR, conn_spec="one_to_one",
                syn_spec={"weight": float(swr_w_sharp_ca3_pyr), "delay": 1.0},
            )
            nest.Connect(
                sw_ripple_ca3, CA3_PYR, conn_spec="one_to_one",
                syn_spec={"weight": float(swr_w_ripple_ca3_pyr), "delay": 1.0},
            )

            _, sw_ripple_ca3_int = make_swr_event_generators(
                n=len(CA3_INT),
                start_ms=swr_start,
                stop_ms=swr_stop,
                sharpwave_rate=0.0,
                ripple_rate_mean=swr_ripple_rate_mean,
                ripple_rate_amp=swr_ripple_rate_amp,
                ripple_hz=swr_ripple_hz,
            )
            nest.Connect(
                sw_ripple_ca3_int, CA3_INT, conn_spec="one_to_one",
                syn_spec={"weight": float(swr_w_ripple_ca3_int), "delay": 1.0},
            )

            sw_sharp_ca1, sw_ripple_ca1_pyr = make_swr_event_generators(
                n=len(CA1_PYR),
                start_ms=swr_start,
                stop_ms=swr_stop,
                sharpwave_rate=swr_sharpwave_rate * 0.6,
                ripple_rate_mean=swr_ripple_rate_mean * 0.6,
                ripple_rate_amp=swr_ripple_rate_amp * 0.6,
                ripple_hz=swr_ripple_hz,
            )
            nest.Connect(
                sw_sharp_ca1, CA1_PYR, conn_spec="one_to_one",
                syn_spec={"weight": float(swr_w_sharp_ca1_pyr), "delay": 1.0},
            )
            nest.Connect(
                sw_ripple_ca1_pyr, CA1_PYR, conn_spec="one_to_one",
                syn_spec={"weight": float(swr_w_ripple_ca1_pyr), "delay": 1.0},
            )

            _, sw_ripple_ca1_ba = make_swr_event_generators(
                n=len(CA1_BASKET),
                start_ms=swr_start,
                stop_ms=swr_stop,
                sharpwave_rate=0.0,
                ripple_rate_mean=swr_ripple_rate_mean * 0.8,
                ripple_rate_amp=swr_ripple_rate_amp * 0.8,
                ripple_hz=swr_ripple_hz,
            )
            nest.Connect(
                sw_ripple_ca1_ba, CA1_BASKET, conn_spec="one_to_one",
                syn_spec={"weight": float(swr_w_ripple_ca1_basket), "delay": 1.0},
            )

    rng = np.random.default_rng(int(seed_connect))

    # Resolve the delayed-copy schedule once: n_delay_copies=1 means "primary
    # delay only" (old behavior); n>1 pulls in the first (n-1) entries of
    # ca3_ee_extra_delays_ms as additional staggered synapses. This list is
    # empty at n=1, so passing it through is a no-op in that case.
    n_delay_copies = max(1, int(ca3_ee_n_delay_copies))
    ca3_ee_extra_delays = list(ca3_ee_extra_delays_ms)[: max(0, n_delay_copies - 1)]

    # --- CA3 E->E: assembly-structured + facilitating (this is the fix) ---
    if ca3_ee_structured:
        assemblies = assign_ca3_assemblies(
            CA3_PYR, n_ca3_assemblies, ca3_assembly_overlap, int(seed_connect)
        )
        id_to_assembly = {
            int(nid): i for i, asm in enumerate(assemblies) for nid in asm
        }

        if ca3_ee_facilitating:
            syn_model = "tsodyks_synapse"
            # Low U + long tau_fac -> facilitation dominates over depression for the
            # first several spikes in a burst, so recurrent drive ramps up rather than
            # decaying between weakly-correlated spikes.
            #
            # History of this parameter (for whoever tunes it next):
            #   1. U=0.15, tau_rec=100 (original): cued-recall ratio inverted below 1
            #      (0.77) -- depression on the cue-driven synapses outlasted the
            #      50ms cue + 300ms measurement window.
            #   2. U=0.05, tau_rec=50 (previous attempt): ratio came back near 1
            #      (0.95), but NOT because completion emerged -- both same- and
            #      other-assembly rates dropped substantially (16-21 Hz -> 12-13 Hz).
            #      U also sets how fast facilitation (u) ramps from a cold start;
            #      lowering it that far meant the ~50ms cue was too short for release
            #      probability to build up at all, weakening transmission broadly
            #      rather than fixing the depression-timing problem specifically.
            #   3. This attempt decouples the two effects: U restored close to the
            #      original (keep real baseline transmission strength) while tau_rec
            #      is cut hard (fast recovery, so any depression clears well before
            #      the post-cue window is measured, instead of lingering into it).
            # Still unverified against real NEST -- confirm with
            # ca3_diagnostics.run_cued_recall_timecourse_probe before trusting this,
            # ideally looking at the actual bin-by-bin shape rather than one more
            # averaged number.
            tsodyks_params = dict(U=0.15, tau_fac=200.0, tau_rec=20.0)
            if ca3_ee_tsodyks_params:
                tsodyks_params.update(ca3_ee_tsodyks_params)
        else:
            syn_model = "static_synapse"
            tsodyks_params = None

        structured_ee_connect(
            CA3_PYR.tolist(),
            id_to_assembly,
            p_within=p_ca3_EE_within,
            p_between=p_ca3_EE_between,
            weight=w_ca3_EE,
            delay=d_fast,
            rng=rng,
            syn_model=syn_model,
            syn_params=tsodyks_params,
            extra_delays=ca3_ee_extra_delays,
            extra_weight_scale=ca3_ee_extra_delay_weight_scale,
        )
    else:
        # legacy flat behavior, kept for A/B comparison against the structured version
        bernoulli_connect(
            CA3_PYR, CA3_PYR, p_ca3_EE, w_ca3_EE, d_fast, rng,
            extra_delays=ca3_ee_extra_delays,
            extra_weight_scale=ca3_ee_extra_delay_weight_scale,
        )

    bernoulli_connect(CA3_PYR, CA3_INT, p_ca3_EI, w_ca3_EI, d_fast, rng)
    bernoulli_connect(CA3_INT, CA3_PYR, p_ca3_IE, w_ca3_IE, d_fast, rng)
    bernoulli_connect(CA3_INT, CA3_INT, p_ca3_II, w_ca3_II, d_fast, rng)

    bernoulli_connect(CA1_PYR, CA1_PYR, p_ca1_EE, w_ca1_EE, d_fast, rng)
    bernoulli_connect(CA1_PYR, CA1_BASKET, p_ca1_EI, w_ca1_EI, d_fast, rng)
    bernoulli_connect(CA1_PYR, CA1_OLM, 0.08, w_ca1_EO, d_slow, rng)
    bernoulli_connect(CA1_BASKET, CA1_PYR, p_ca1_IE, w_ca1_IE, d_fast, rng)
    bernoulli_connect(CA1_BASKET, CA1_BASKET, 0.10, w_ca1_II, d_fast, rng)
    bernoulli_connect(CA1_OLM, CA1_PYR, p_ca1_OE, w_ca1_OE, d_slow, rng)

    bernoulli_connect(CA3_PYR, CA1_PYR, p_schaffer_pyr, w_schaffer_pyr, d_slow, rng)
    bernoulli_connect(CA3_PYR, CA1_BASKET, p_schaffer_basket, w_schaffer_ba, d_fast, rng)

    spk_ca1_pyr = nest.Create("spike_recorder")
    spk_ca1_ba = nest.Create("spike_recorder")
    spk_ca1_olm = nest.Create("spike_recorder")
    nest.Connect(CA1_PYR, spk_ca1_pyr)
    nest.Connect(CA1_BASKET, spk_ca1_ba)
    nest.Connect(CA1_OLM, spk_ca1_olm)

    spk_ca3_pyr = nest.Create("spike_recorder")
    spk_ca3_int = nest.Create("spike_recorder")
    nest.Connect(CA3_PYR, spk_ca3_pyr)
    nest.Connect(CA3_INT, spk_ca3_int)

    try:
        vm = nest.Create("multimeter", params={"record_from": ["V_m", "U_m"], "interval": 0.2})
    except Exception:
        vm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.2})

    ca1_probe = CA1_PYR[:5]
    ca3_probe = CA3_PYR[:5]
    nest.Connect(vm, ca1_probe)
    nest.Connect(vm, ca3_probe)

    print("\nConnectivity stats:")
    conn_stats("CA3E->CA3E", CA3_PYR, CA3_PYR)
    conn_stats("CA3E->CA3I", CA3_PYR, CA3_INT)
    conn_stats("CA3I->CA3E", CA3_INT, CA3_PYR)
    conn_stats("Sch CA3E->CA1E", CA3_PYR, CA1_PYR)
    conn_stats("Sch CA3E->CA1I", CA3_PYR, CA1_BASKET)
    conn_stats("CA1E->CA1E", CA1_PYR, CA1_PYR)
    conn_stats("CA1I->CA1E", CA1_BASKET, CA1_PYR)

    if ca3_ee_structured:
        # within- vs between-assembly breakdown, so you can actually verify structure
        # (not just density) from the run itself instead of trusting the flat summary.
        # NOTE: dedupe by (source, target) before counting -- with delay-copies on,
        # GetConnections() returns up to n_delay_copies separate synapse objects for
        # the SAME logical pair, so counting rows directly overcounts by that factor
        # and can push p_eff above 1 (a nonsense value for a probability).
        conns = nest.GetConnections(CA3_PYR, CA3_PYR)
        src = np.atleast_1d(conns.get("source"))
        tgt = np.atleast_1d(conns.get("target"))
        pairs = set(zip(src.tolist(), tgt.tolist()))
        within = sum(1 for s, t in pairs if id_to_assembly.get(int(s)) == id_to_assembly.get(int(t)))
        between = len(pairs) - within
        n_pairs_within = sum(
            len(a) * (len(a) - 1) for a in assemblies
        )
        n_pairs_between = int(N_ca3_pyr) * (int(N_ca3_pyr) - 1) - n_pairs_within
        print(
            f"CA3E->CA3E within-assembly : {within:6d} / {n_pairs_within:6d} possible "
            f"(p_eff={within/max(n_pairs_within,1):.3f}) "
            f"[{len(src)} raw synapse objects -> {len(pairs)} distinct pairs, "
            f"up to {n_delay_copies}x delay-copies per connected pair]"
        )
        print(
            f"CA3E->CA3E between-assembly: {between:6d} / {n_pairs_between:6d} possible "
            f"(p_eff={between/max(n_pairs_between,1):.3f})"
        )

    return dict(
        PYR=CA1_PYR,
        BASKET=CA1_BASKET,
        OLM=CA1_OLM,
        spk_pyr=spk_ca1_pyr,
        spk_ba=spk_ca1_ba,
        spk_olm=spk_ca1_olm,
        CA3_PYR=CA3_PYR,
        CA3_INT=CA3_INT,
        spk_ca3_pyr=spk_ca3_pyr,
        spk_ca3_int=spk_ca3_int,
        vm=vm,
        ca1_probe=ca1_probe,
        ca3_probe=ca3_probe,
        theta_on=bool(theta_on),
        theta_hz=float(theta_hz),
        swr_on=bool(swr_on),
        swr_events=list(swr_events) if swr_on and swr_events is not None else [],
        swr_ripple_hz=float(swr_ripple_hz),
        ca3_assemblies=assemblies if ca3_ee_structured else None,
        ca3_id_to_assembly=id_to_assembly if ca3_ee_structured else None,
    )
