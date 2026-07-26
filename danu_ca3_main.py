"""
ca3_danu_main_parallel.py
--------------------------
Same pipeline as ca3_danu_main.py, but the 96 simulations run across
multiple OS processes instead of one at a time. NEST's own multithreading
(local_num_threads) gives sub-linear speedup and only parallelizes WITHIN
one simulation; on a single machine you get much better throughput running
many single-threaded simulations concurrently than fewer multi-threaded
ones back to back.

"""

import os
import numpy as np
import nest
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from tinyhippo import build_ca1_ca3_izh
from danu_cork_eeg import (
    run_grade_experiment, summarise, print_summary_table, plot_grade_comparison,
    NUM_SEEDS, SAVE_DIR,
)

os.makedirs(SAVE_DIR, exist_ok=True)

GRADE_TO_P_CA3_EE = {1: 0.02, 2: 0.035, 3: 0.05, 4: 0.07}
N_SUBJECTS_PER_GRADE = 5
SIM_MS = 60000.0
N_CHANNELS = 8
THREADS_PER_SIM = 1                         # NEST threads per worker process
N_WORKERS = max(1, (os.cpu_count() or 8) - 1)  # leave 1 core free; bump toward full count if desktop stays responsive

CACHE_DIR = os.path.join(SAVE_DIR, "ca3_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _subject_cache_path(grade, subj):
    p_ee = GRADE_TO_P_CA3_EE[grade]
    return os.path.join(
        CACHE_DIR,
        f"grade{grade}_pEE{p_ee}_sim{int(SIM_MS)}_ch{N_CHANNELS}_subj{subj}.npz",
    )


def add_ca3_channel_probe(net, n_channels, seed):
    CA3_PYR = net["CA3_PYR"]
    rng = np.random.default_rng(seed)
    ids = np.array(CA3_PYR.tolist())
    rng.shuffle(ids)
    net["ca3_channel_groups"] = np.array_split(ids, n_channels)
    vm_ca3 = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 5.0})  # 200 Hz
    nest.Connect(vm_ca3, CA3_PYR)
    net["vm_ca3_full"] = vm_ca3
    return net


def extract_pseudo_eeg(net, n_channels, drop_ms=200.0):
    ev = nest.GetStatus(net["vm_ca3_full"], "events")[0]
    times = np.asarray(ev["times"])
    senders = np.asarray(ev["senders"])
    V = np.asarray(ev["V_m"])
    keep = times >= drop_ms
    times, senders, V = times[keep], senders[keep], V[keep]
    t_grid = np.unique(times)
    t_grid.sort()
    channels = np.zeros((n_channels, len(t_grid)), dtype=np.float32)
    for c, group in enumerate(net["ca3_channel_groups"]):
        mask = np.isin(senders, group)
        gt, gv = times[mask], V[mask]
        idx = np.searchsorted(t_grid, gt)
        sums = np.zeros(len(t_grid))
        counts = np.zeros(len(t_grid))
        np.add.at(sums, idx, gv)
        np.add.at(counts, idx, 1)
        counts[counts == 0] = 1
        channels[c] = sums / counts
    channels = (channels - channels.mean(axis=1, keepdims=True)) / (
        channels.std(axis=1, keepdims=True) + 1e-8
    )
    return channels

def simulate_one_subject(grade, subj):
    """Runs inside its own worker process (own Python interpreter, own
    NEST kernel -- NEST's global state is not fork-safe, so the pool below
    uses the 'spawn' start method to guarantee a clean process, not a fork).
    """
    path = _subject_cache_path(grade, subj)
    if os.path.exists(path):
        return grade, subj, path, True  # already cached

    import time
    import traceback

    time.sleep(np.random.default_rng(1000 * grade + subj).uniform(0, 2.0))

    p_ee = GRADE_TO_P_CA3_EE[grade]
    seed = 1000 * grade + subj

    try:
        os.environ["OMP_NUM_THREADS"] = "1"  # avoid BLAS thread oversubscription
        nest.SetKernelStatus({"local_num_threads": THREADS_PER_SIM})

        net = build_ca1_ca3_izh(
            p_ca3_EE=p_ee,
            theta_on=True,
            swr_on=True,
            swr_events=[(t, t + 80.0) for t in np.arange(2000.0, SIM_MS, 4000.0)],
            seed_connect=seed,
        )
        net = add_ca3_channel_probe(net, N_CHANNELS, seed)
        nest.Simulate(float(SIM_MS))
        rec = extract_pseudo_eeg(net, N_CHANNELS)

        np.savez_compressed(path, rec=rec)
        return grade, subj, path, False

    except Exception as e:
        # NEST's own exception types (e.g. KernelException) aren't
        # reliably picklable across the ProcessPoolExecutor boundary --
        # shipping them back to the parent raises a *separate* pickling
        # error that masks the real failure and kills the whole pool.
        # Stringify here instead so the real message survives.
        err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return grade, subj, err_msg, "ERROR"


MAX_POOL_RESTARTS = 20  # generous -- each restart just re-pools whatever's left


def load_ca3_grade_data():
    """Runs the subject simulations, resilient to a worker segfault.

    A segfault kills that OS process outright -- Python's try/except cannot
    catch it. What DOES happen is that any pending future submitted to that
    same ProcessPoolExecutor raises BrokenProcessPool as soon as you call
    .result() on it, which would otherwise abort the WHOLE batch (not just
    the crashed subject). We catch that at the pool level, throw the dead
    pool away, and start a fresh one for whatever subjects are still missing
    from the on-disk cache -- so a crash costs you at most the in-flight
    tasks from that one pool generation, not the whole run.
    """
    tasks = [
        (grade, subj)
        for grade in GRADE_TO_P_CA3_EE
        for subj in range(N_SUBJECTS_PER_GRADE)
    ]

    restarts = 0
    while True:
        todo = [t for t in tasks if not os.path.exists(_subject_cache_path(*t))]
        if not todo:
            break

        print(f"[Data] {len(tasks) - len(todo)}/{len(tasks)} subjects cached, "
              f"{len(todo)} left, {N_WORKERS} workers "
              f"({THREADS_PER_SIM} NEST thread(s) each) [pool attempt {restarts + 1}]")

        ctx = mp.get_context("spawn")  # avoid forking a process with NEST already loaded
        try:
            with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
                futures = {ex.submit(simulate_one_subject, g, s): (g, s) for g, s in todo}
                done_count = 0
                for fut in as_completed(futures):
                    g, s = futures[fut]
                    fut.result()  # raises BrokenProcessPool for the WHOLE pool if any worker died
                    done_count += 1
                    print(f"[Data] grade {g} subject {s + 1}/{N_SUBJECTS_PER_GRADE} done "
                          f"({done_count}/{len(todo)} this pass)")
        except BrokenProcessPool:
            restarts += 1
            print(f"[Data] Worker pool crashed (likely a NEST segfault in one worker) -- "
                  f"restarting pool, resuming from cache (restart {restarts}/{MAX_POOL_RESTARTS})")
            if restarts >= MAX_POOL_RESTARTS:
                print("[Data] Too many pool restarts -- stopping. Rerun the script later to resume "
                      "(everything already cached will be skipped).")
                break
            continue
        except Exception as e:
            print(f"[Data] Unexpected error in this pass, restarting pool: {e}")
            restarts += 1
            if restarts >= MAX_POOL_RESTARTS:
                break
            continue

    grade_data = {g: [] for g in GRADE_TO_P_CA3_EE}
    missing = []
    for grade, subj in tasks:
        path = _subject_cache_path(grade, subj)
        if not os.path.exists(path):
            missing.append((grade, subj))
            continue
        with np.load(path) as data:
            grade_data[grade].append(data["rec"])
    if missing:
        print(f"[Data] WARNING: {len(missing)} subjects still missing: {missing}")
    return grade_data


def main():
    grade_data = load_ca3_grade_data()

    grade_results = {}
    for grade, recordings in grade_data.items():
        if len(recordings) < 3:
            print(f"  [SKIP] Grade {grade} has only {len(recordings)} recordings, skipping")
            continue
        metrics = run_grade_experiment(grade, recordings, n_seeds=NUM_SEEDS)
        if metrics is None:
            continue
        summary, vals = summarise(metrics)
        grade_results[grade] = {"metrics": metrics, "summary": summary, "vals": vals}

    if not grade_results:
        print("\n[ERROR] No grades had sufficient data.")
        return

    print_summary_table(grade_results)
    plot_path = os.path.join(SAVE_DIR, "ca3_danu_comparison.png")
    plot_grade_comparison(grade_results, save_path=plot_path)
    print(f"\n[Done] Results saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
