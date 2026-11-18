#!/usr/bin/env python3
"""
CA1-like microcircuit in NEST using *only* Izhikevich neurons + spike generators as inputs.

Key points (per NEST izhikevich model):
- Incoming spikes change V_m directly by the synaptic weight (no PSC kernel inside this neuron model).
- So weights are in mV “jumps” and can be much smaller than iaf_psc_exp currents.

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


def bernoulli_connect(pre, post, p, weight, delay, rng):
    """Explicit Bernoulli connectivity; easy to hack for motif studies."""
    pre_ids = np.array(pre, dtype=int)
    post_ids = np.array(post, dtype=int)
    for src in pre_ids:
        mask = rng.random(post_ids.size) < p
        targets = post_ids[mask].tolist()
        if targets:
            nest.Connect([int(src)], targets, syn_spec={"weight": float(weight), "delay": float(delay)})


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
):
    nest.ResetKernel()
    nest.SetKernelStatus(
        {
            "resolution": 0.1,        # ms
            "local_num_threads": 4,   # tune for your CPU
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

    # Vm traces (plot per neuron to avoid “diagonal ramps” artifact)
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
    rate_ca3_drive_pyr=600.0,
    rate_ec_ca3_pyr=800.0,
    rate_dg_ca3_pyr=1000.0,
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
    # --- RNG seed
    seed_connect=42,
):
    """A compact CA1+CA3 toy network in the same style as build_ca1_izh.

    CA3 is modeled as a recurrent excitatory (pyramidal) + inhibitory (basket-like) network.
    CA3 exc projects to CA1 PYR and CA1 BASKET via Schaffer-collateral-like connections.

    Notes:
      - Uses NEST 'izhikevich' only; synaptic weights are direct V_m jumps in mV.
      - This is *not* a full CA3 microanatomy (mossy fibers, multiple interneuron classes, etc.).
        It's the minimal recurrent E/I core you can later refine.
    """

    nest.ResetKernel()
    nest.SetKernelStatus(
        {
            "resolution": 0.1,
            "local_num_threads": 4,
            "print_time": True,
            "overwrite_files": True,
        }
    )
    safe_set_seeds()

    if "izhikevich" not in nest.Models("nodes"):
        raise RuntimeError("NEST model 'izhikevich' not found in this installation.")

    # --- Izhikevich parameter sets
    pyr_params = dict(a=0.02, b=0.2, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    basket_params = dict(a=0.1, b=0.2, c=-65.0, d=2.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    olm_params = dict(a=0.02, b=0.25, c=-65.0, d=2.0, V_m=-65.0, U_m=-16.25, I_e=0.0)

    # --- Populations
    CA1_PYR = nest.Create("izhikevich", int(N_ca1_pyr), params=pyr_params)
    CA1_BASKET = nest.Create("izhikevich", int(N_ca1_basket), params=basket_params)
    CA1_OLM = nest.Create("izhikevich", int(N_ca1_olm), params=olm_params)

    CA3_PYR = nest.Create("izhikevich", int(N_ca3_pyr), params=pyr_params)
    CA3_INT = nest.Create("izhikevich", int(N_ca3_int), params=basket_params)

    # --- External inputs (independent poisson per neuron)
    # CA1 receives ECIII-like context drive + some unspecific drive to basket cells
    ECIII_to_CA1 = nest.Create("poisson_generator", len(CA1_PYR), params={"rate": float(rate_ec_ca1_pyr)})
    DRIVE_to_CA1_BA = nest.Create(
        "poisson_generator", len(CA1_BASKET), params={"rate": float(rate_drive_ca1_basket)}
    )

    # CA3 receives ECII/EC input + DG-like drive + a moderate background drive
    EC_to_CA3 = nest.Create("poisson_generator", len(CA3_PYR), params={"rate": float(rate_ec_ca3_pyr)})
    DG_to_CA3 = nest.Create("poisson_generator", len(CA3_PYR), params={"rate": float(rate_dg_ca3_pyr)})
    DRIVE_to_CA3 = nest.Create("poisson_generator", len(CA3_PYR), params={"rate": float(rate_ca3_drive_pyr)})
    DRIVE_to_CA3_INT = nest.Create(
        "poisson_generator", len(CA3_INT), params={"rate": float(rate_drive_ca3_int)}
    )

    # --- Synapse weights (mV jumps)
    # external -> excitatory
    w_ec = 2.0
    w_dg = 3.0
    w_drive = 2.0

    # CA3 recurrent
    w_ca3_EE = 0.9
    w_ca3_EI = 1.6
    w_ca3_IE = -5.5
    w_ca3_II = -4.5

    # CA1 local (as in build_ca1_izh)
    w_ca1_EE = 0.8
    w_ca1_EI = 1.5
    w_ca1_EO = 1.2
    w_ca1_IE = -5.0
    w_ca1_II = -4.0
    w_ca1_OE = -3.0

    # Schaffer collateral CA3 -> CA1
    w_schaffer_pyr = 2.4
    w_schaffer_ba = 2.8

    d_fast = 1.5
    d_slow = 3.0

    # --- Wire inputs one-to-one
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


    # --- Theta rhythm (optional)
    if theta_on:
        # CA3 theta drive
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

        # CA1 theta drive
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

    # --- SWR (sharp-wave ripple) events: brief CA3-driven bursts that can recruit CA1
    if swr_on:
        if swr_events is None:
            # default: one SWR around the middle of a 1 s sim
            swr_events = [(550.0, 630.0)]  # (start_ms, stop_ms)

        for (swr_start, swr_stop) in swr_events:
            # CA3: sharp wave mainly excites PYR; ripple mainly drives INH (and weakly PYR)
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
            # Ripple to CA3 PYR (weak)
            nest.Connect(
                sw_ripple_ca3, CA3_PYR, conn_spec="one_to_one",
                syn_spec={"weight": float(swr_w_ripple_ca3_pyr), "delay": 1.0},
            )

            # Separate ripple drive for CA3 interneurons (stronger)
            _, sw_ripple_ca3_int = make_swr_event_generators(
                n=len(CA3_INT),
                start_ms=swr_start,
                stop_ms=swr_stop,
                sharpwave_rate=0.0,  # no sharpwave to INH in this toy version
                ripple_rate_mean=swr_ripple_rate_mean,
                ripple_rate_amp=swr_ripple_rate_amp,
                ripple_hz=swr_ripple_hz,
            )
            nest.Connect(
                sw_ripple_ca3_int, CA3_INT, conn_spec="one_to_one",
                syn_spec={"weight": float(swr_w_ripple_ca3_int), "delay": 1.0},
            )

            # CA1: sharpwave weakly excites PYR; ripple strongly drives basket, weakly PYR
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

            # Ripple to CA1 basket cells (strong)
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

    # --- Recurrent connectivity (explicit Bernoulli)
    rng = np.random.default_rng(int(seed_connect))

    # CA3 recurrent core
    bernoulli_connect(CA3_PYR, CA3_PYR, p_ca3_EE, w_ca3_EE, d_fast, rng)       # CA3 E->E
    bernoulli_connect(CA3_PYR, CA3_INT, p_ca3_EI, w_ca3_EI, d_fast, rng)       # CA3 E->I
    bernoulli_connect(CA3_INT, CA3_PYR, p_ca3_IE, w_ca3_IE, d_fast, rng)       # CA3 I->E
    bernoulli_connect(CA3_INT, CA3_INT, p_ca3_II, w_ca3_II, d_fast, rng)       # CA3 I->I

    # CA1 local microcircuit (same motifs as original)
    bernoulli_connect(CA1_PYR, CA1_PYR, p_ca1_EE, w_ca1_EE, d_fast, rng)
    bernoulli_connect(CA1_PYR, CA1_BASKET, p_ca1_EI, w_ca1_EI, d_fast, rng)
    bernoulli_connect(CA1_PYR, CA1_OLM, 0.08, w_ca1_EO, d_slow, rng)
    bernoulli_connect(CA1_BASKET, CA1_PYR, p_ca1_IE, w_ca1_IE, d_fast, rng)
    bernoulli_connect(CA1_BASKET, CA1_BASKET, 0.10, w_ca1_II, d_fast, rng)
    bernoulli_connect(CA1_OLM, CA1_PYR, p_ca1_OE, w_ca1_OE, d_slow, rng)

    # --- CA3 -> CA1 (Schaffer collaterals): excitatory to CA1 pyramidal and basket
    bernoulli_connect(CA3_PYR, CA1_PYR, p_schaffer_pyr, w_schaffer_pyr, d_slow, rng)
    bernoulli_connect(CA3_PYR, CA1_BASKET, p_schaffer_basket, w_schaffer_ba, d_fast, rng)

    # --- Recorders
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

    # Vm probes (few neurons in CA1 + CA3)
    try:
        vm = nest.Create("multimeter", params={"record_from": ["V_m", "U_m"], "interval": 0.2})
    except Exception:
        vm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.2})

    ca1_probe = CA1_PYR[:5]
    ca3_probe = CA3_PYR[:5]
    nest.Connect(vm, ca1_probe)
    nest.Connect(vm, ca3_probe)

    # Connectivity stats
    print("\nConnectivity stats:")
    conn_stats("CA3E->CA3E", CA3_PYR, CA3_PYR)
    conn_stats("CA3E->CA3I", CA3_PYR, CA3_INT)
    conn_stats("CA3I->CA3E", CA3_INT, CA3_PYR)
    conn_stats("Sch CA3E->CA1E", CA3_PYR, CA1_PYR)
    conn_stats("Sch CA3E->CA1I", CA3_PYR, CA1_BASKET)
    conn_stats("CA1E->CA1E", CA1_PYR, CA1_PYR)
    conn_stats("CA1I->CA1E", CA1_BASKET, CA1_PYR)

    return dict(
        # CA1
        PYR=CA1_PYR,
        BASKET=CA1_BASKET,
        OLM=CA1_OLM,
        spk_pyr=spk_ca1_pyr,
        spk_ba=spk_ca1_ba,
        spk_olm=spk_ca1_olm,
        # CA3
        CA3_PYR=CA3_PYR,
        CA3_INT=CA3_INT,
        spk_ca3_pyr=spk_ca3_pyr,
        spk_ca3_int=spk_ca3_int,
        # probes
        vm=vm,
        ca1_probe=ca1_probe,
        ca3_probe=ca3_probe,
        # metadata (for plotting / debugging)
        theta_on=bool(theta_on),
        theta_hz=float(theta_hz),
        swr_on=bool(swr_on),
        swr_events=list(swr_events) if swr_on and swr_events is not None else [],
        swr_ripple_hz=float(swr_ripple_hz),
    )


def run_report_plot(net, sim_ms=1000.0):
    nest.Simulate(float(sim_ms))

    # Spike counts + mean rates
    ev_p = nest.GetStatus(net["spk_pyr"], "events")[0]
    ev_b = nest.GetStatus(net["spk_ba"], "events")[0]
    ev_o = nest.GetStatus(net["spk_olm"], "events")[0]

    print("\nSpike counts:")
    print("  PYR   :", len(ev_p["times"]))
    print("  BASKET:", len(ev_b["times"]))
    print("  OLM   :", len(ev_o["times"]))

    print("\nMean firing rates (Hz):")
    print(f"  PYR   : {mean_rate(net['PYR'], net['spk_pyr'], sim_ms):.2f}")
    print(f"  BASKET: {mean_rate(net['BASKET'], net['spk_ba'], sim_ms):.2f}")
    print(f"  OLM   : {mean_rate(net['OLM'], net['spk_olm'], sim_ms):.2f}")

    # Optional CA3 report (present when using build_ca1_ca3_izh)
    if "CA3_PYR" in net:
        ev_c3e = nest.GetStatus(net["spk_ca3_pyr"], "events")[0]
        ev_c3i = nest.GetStatus(net["spk_ca3_int"], "events")[0]
        print("\nCA3 Spike counts:")
        print("  CA3_PYR:", len(ev_c3e["times"]))
        print("  CA3_INT:", len(ev_c3i["times"]))
        print("\nCA3 Mean firing rates (Hz):")
        print(f"  CA3_PYR: {mean_rate(net['CA3_PYR'], net['spk_ca3_pyr'], sim_ms):.2f}")
        print(f"  CA3_INT: {mean_rate(net['CA3_INT'], net['spk_ca3_int'], sim_ms):.2f}")

    # Plots
    import matplotlib.pyplot as plt

    def raster(spk, title):
        ev = nest.GetStatus(spk, "events")[0]
        plt.figure()
        plt.plot(ev["times"], ev["senders"], ".")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID")
        plt.title(title)

    raster(net["spk_pyr"], "CA1 PYR spikes (izhikevich)")
    raster(net["spk_ba"], "CA1 Basket spikes (izhikevich)")
    raster(net["spk_olm"], "CA1 OLM spikes (izhikevich)")

    if "CA3_PYR" in net:
        raster(net["spk_ca3_pyr"], "CA3 PYR spikes (izhikevich)")
        raster(net["spk_ca3_int"], "CA3 INH spikes (izhikevich)")

    # --- Population rate traces (this is where theta/SWR become obvious)
    def get_times(spk):
        ev = nest.GetStatus(spk, "events")[0]
        return np.asarray(ev["times"], dtype=float)

    def binned_rate(times_ms, t_stop_ms, n_cells, bin_ms):
        edges = np.arange(0.0, t_stop_ms + bin_ms, bin_ms)
        counts, _ = np.histogram(times_ms, bins=edges)
        rate_hz = counts / (bin_ms / 1000.0) / max(int(n_cells), 1)
        centers = edges[:-1] + bin_ms / 2.0
        return centers, rate_hz

    def shade_swr(ax):
        if net.get("swr_on", False):
            for (s, e) in net.get("swr_events", []):
                ax.axvspan(float(s), float(e), alpha=0.15)

    # Low-res (sharp wave envelope)
    bin_env = 10.0  # ms
    plt.figure()
    t, r = binned_rate(get_times(net["spk_pyr"]), sim_ms, len(net["PYR"]), bin_env)
    plt.plot(t, r, label="CA1 PYR")
    if "CA3_PYR" in net:
        t3, r3 = binned_rate(get_times(net["spk_ca3_pyr"]), sim_ms, len(net["CA3_PYR"]), bin_env)
        plt.plot(t3, r3, label="CA3 PYR")
    plt.xlabel("Time (ms)")
    plt.ylabel(f"Rate (Hz), {bin_env:g} ms bins")
    plt.title("Population rate (envelope) with SWR window")
    ax = plt.gca()
    shade_swr(ax)
    plt.legend()

    # High-res (ripple band visibility)
    bin_rip = 2.0  # ms ~ can resolve ~180 Hz structure
    plt.figure()
    if "CA3_PYR" in net:
        t3i, r3i = binned_rate(get_times(net["spk_ca3_int"]), sim_ms, len(net["CA3_INT"]), bin_rip)
        plt.plot(t3i, r3i, label="CA3 INH")
    t1b, r1b = binned_rate(get_times(net["spk_ba"]), sim_ms, len(net["BASKET"]), bin_rip)
    plt.plot(t1b, r1b, label="CA1 Basket")
    plt.xlabel("Time (ms)")
    plt.ylabel(f"Rate (Hz), {bin_rip:g} ms bins")
    plt.title("Fast population rate (ripple-scale) with SWR window")
    ax = plt.gca()
    shade_swr(ax)
    plt.legend()

    # Vm trace per neuron (critical to avoid fake diagonals)
    ev = nest.GetStatus(net["vm"], "events")[0]
    times = np.array(ev["times"])
    senders = np.array(ev["senders"])
    V = np.array(ev["V_m"])

    plt.figure()
    for gid in np.unique(senders):
        m = (senders == gid)
        idx = np.argsort(times[m])
        plt.plot(times[m][idx], V[m][idx])
    plt.xlabel("Time (ms)")
    plt.ylabel("V_m (mV)")
    plt.title("PYR membrane traces (per neuron)")
    plt.show()


if __name__ == "__main__":
    net = build_ca1_ca3_izh(
        # CA1
        N_ca1_pyr=200,
        N_ca1_basket=15,
        N_ca1_olm=12,
        # CA3
        N_ca3_pyr=300,
        N_ca3_int=41,
        # Drive (keep these as your first tuning knobs)
        rate_ec_ca1_pyr=630.0,
        rate_dg_ca3_pyr=840.0,
        rate_ec_ca3_pyr=560.0,
        rate_ca3_drive_pyr=420.0,
        rate_drive_ca1_basket=840.0,
        rate_drive_ca3_int=840.0,
        # Theta (optional)
        theta_on=True,
        theta_hz=8.0,
        theta_rate_mean_ca3=1200.0,
        theta_rate_amp_ca3=1100.0,
        theta_rate_mean_ca1=1200.0,
        theta_rate_amp_ca1=1100.0,
        # SWR (optional)
        swr_on=True,
        swr_events=[(550.0, 630.0)],
        swr_ripple_hz=180.0,
        swr_ripple_rate_mean=1200.0,
        swr_ripple_rate_amp=900.0,
    )
    run_report_plot(net, sim_ms=1000.0)

"""
Fast tuning cheats (because biology is rude):
- Too quiet (no spikes): increase rate_ca3_pyr and/or w_ca3_pyr (e.g., 3.0 -> 4.0).
- PYR too hot: make w_basket_pyr more negative (e.g., -5.0 -> -7.0) or increase p_IE.
- Basket silent but PYR active: increase rate_ca3_ba or w_ca3_basket.
"""