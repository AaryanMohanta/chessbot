"""Shared CPU-affinity-pinning and concurrency-safety helpers for local
self-play testing (ratings/run_variant_round_robin.py, ratings/sprt.py).

Centralized here after the SAME bug was found independently in both
callers: naive consecutive-index pairing (white=core 2i, black=core 2i+1)
for each concurrent game/pair slot. On a uniform-core machine that's
merely suboptimal; on this dev box it was actively harmful.

Hardware (Intel Core Ultra 7 165U, a hybrid part): 12 physical cores, 14
logical threads -- only 2 P-cores have hyperthreading (4 threads: logical
0-3, HT-sibling pairs (0,1) and (2,3)), the other 10 physical cores are
single-thread E-cores (8 plain E-cores at logical 4-11, 2 low-power
E-cores at logical 12-13). Naive (2*i, 2*i+1) pairing put one slot's
white+black on the SAME P-core's two HT siblings (harmless) but every
other slot entirely on raw E-cores -- and E-cores here aren't just slower
to compile on, they run the search itself at a fraction of normal speed
(observed directly: several concurrency=6 calibration games compiled fine
then lost on a time flag at 30-70 plies against a 10s+100ms clock). Worse,
mixing a P-core game with E-core games in the same batch doesn't just risk
more crashes, it biases whatever comparison is running: whichever
candidate/variant happens to land on the fast cores that game gets a
speed edge unrelated to its actual strength.

PREFERRED_CORE_PAIRS fixes this two ways: (1) both cores of a pair are
always the same class, so no single game is internally lopsided, and (2)
HT-sibling pairs and the 2 low-power E-cores are excluded entirely --
least reliable in calibration -- capping safe concurrency at 5 on this
machine rather than the naive 7 (14 logical / 2). This is hardcoded to
one dev laptop's topology, not portable, and has nothing to do with the
real competition's own "1 dedicated core" constraint on different
reference hardware -- it only matters for making LOCAL self-play testing
trustworthy.

Memory: a second, independent failure mode found in the same session --
concurrent numba/LLVM compiles are memory-hungry, and this laptop
routinely runs with well under half its 32GB free (browser tabs, Discord,
sometimes a game) even with no testing running. Concurrency=2 with
correctly-pinned distinct P-cores still failed 100% once when free RAM
was ~7GB; the same settings were reliable once free RAM recovered to
~10GB+. So concurrency has to respect current memory headroom, not just
CPU topology -- and that headroom changes independently of anything this
process does, so it must be (re-)checked at the start of each run, not
hardcoded once.
"""
from __future__ import annotations

PREFERRED_CORE_PAIRS: list[tuple[int, int]] = [(0, 2), (4, 5), (6, 7), (8, 9), (10, 11)]

# Empirically, each concurrently-compiling agent process's peak RSS during
# JIT compilation was observed in the 0.5-1.5GB range on this build (numba
# + llvmlite + numpy + the search module's own arrays) -- 1.5GB/process,
# 3GB/slot (white+black), is a deliberately conservative per-slot budget
# so a probe-free static estimate still errs toward "too cautious" rather
# than reproducing the 100%-failure runs from under-provisioning.
GB_PER_SLOT = 3.0

MIN_FREE_GB_FOR_ANY_CONCURRENCY = 5.0  # below this, don't even trust concurrency=1 pinning; go fully unpinned+sequential


def free_ram_gb() -> float | None:
    """None if psutil isn't installed or the query fails -- callers should
    treat that as "unknown, assume the worst" rather than crash."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return None


def safe_concurrency(requested: int, verbose: bool = True) -> tuple[int, bool]:
    """Returns (concurrency, use_cpu_pin) after capping `requested` against
    both this machine's known-safe core pairs AND its current free RAM.
    Memory is checked fresh on every call (not cached) since it fluctuates
    with whatever else the user is running, independent of this process.
    """
    max_by_topology = len(PREFERRED_CORE_PAIRS)
    concurrency = max(1, min(requested, max_by_topology))

    free_gb = free_ram_gb()
    if free_gb is None:
        if verbose:
            print("NOTE: couldn't read free RAM (psutil unavailable?) -- proceeding without a memory-based cap.", flush=True)
        return concurrency, True

    if free_gb < MIN_FREE_GB_FOR_ANY_CONCURRENCY:
        if verbose:
            print(f"NOTE: only {free_gb:.1f}GB free RAM (< {MIN_FREE_GB_FOR_ANY_CONCURRENCY:.0f}GB floor) -- "
                  f"falling back to fully sequential, unpinned (concurrency=1). Close some background "
                  f"apps and rerun for more throughput.", flush=True)
        return 1, False

    max_by_memory = max(1, int(free_gb // GB_PER_SLOT))
    if max_by_memory < concurrency:
        if verbose:
            print(f"NOTE: capping concurrency {concurrency} -> {max_by_memory} "
                  f"({free_gb:.1f}GB free / {GB_PER_SLOT:.0f}GB per concurrent game slot).", flush=True)
        concurrency = max_by_memory

    return concurrency, True
