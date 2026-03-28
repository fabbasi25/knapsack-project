# ============================================================
# tests_p.py — Test harness for QTG knapsack solver
# ============================================================
#
# Identical structure to the original QAOA test file, but
# imports run_knapsack from qtg instead of qaoa.
#
# The QAOA parameter 'p' (circuit depth) maps to 'k' (number
# of Grover iterations) in the QTG algorithm.
#
# Usage:
#   python tests_p.py <output_file> <n_items>
#
# When run as a SLURM array job, each task gets a different
# random seed via SLURM_ARRAY_TASK_ID.

import os
import sys
import time
import numpy as np
from itertools import product

print("top of file", flush=True)

# ---- swap this single line to switch between solvers ----
from qtg import run_knapsack
from qiskit_aer.primitives import Sampler as AerSampler

sampler = AerSampler()


# ============================================================
# Classical references
# ============================================================

def brute_force_multiknapsack(weights, values, capacities, time_limit=None):
    """Exact solver by exhaustive enumeration.

    For a single knapsack (m=1) this iterates over all 2^n subsets.
    A time_limit (seconds) can be set so large instances don't hang.
    """
    n = len(weights)
    m = len(capacities)
    best_value = 0
    best_x = None
    start_time = time.time()

    for x in product([0, 1], repeat=n * m):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            print(f"Time limit {time_limit}s reached, stopping brute-force")
            break

        x_arr = np.array(x).reshape(m, n)
        if np.any(x_arr.sum(axis=0) > 1):
            continue
        if np.all(x_arr @ weights <= capacities):
            val = np.sum(x_arr * values)
            if val > best_value:
                best_value = val
                best_x = x_arr

    return best_x, best_value


def greedy_knapsack(weights, values, capacities):
    """Fast greedy heuristic used as the reference for large instances."""
    n = len(weights)
    m = len(capacities)
    best_x = np.zeros((m, n), dtype=int)
    remaining = capacities.copy()

    ratio = [v / w for v, w in zip(values, weights)]
    order = np.argsort(ratio)[::-1]

    for j in order:
        for i in range(m):
            if weights[j] <= remaining[i]:
                best_x[i, j] = 1
                remaining[i] -= weights[j]
                break

    total_value = np.sum(best_x * values)
    return best_x, total_value


# ============================================================
# Random instance generator
# ============================================================

def generate_knapsack(n, weight_range=(1, 50), value_range=(1, 100),
                      capacity_ratio=0.5, seed=None):
    """Generate a random n-item single-knapsack instance.

    Returns (weights, values, [capacity]) — the same format that
    run_knapsack expects.
    """
    rng = np.random.default_rng(seed)

    weights = rng.integers(*weight_range, size=n)
    values  = rng.integers(*value_range, size=n)

    capacity = int(capacity_ratio * np.sum(weights))

    return weights.tolist(), values.tolist(), [capacity]


# ============================================================
# Main test loop
# ============================================================

def run_p_tests(output_file, n):
    """For a random n-item instance, run QTG at several Grover
    iteration counts (k = 1..7) and compare against a classical
    reference (brute-force for n<=20, greedy otherwise).

    Results are written as CSV lines:  k, qtg_value, reference_value
    """
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    weights, values, capacities = generate_knapsack(n=n, seed=task_id)

    print(f"Running task {task_id}", flush=True)

    # choose classical reference
    if n <= 20:
        print("Running brute-force as reference")
        reference_x, reference_value = brute_force_multiknapsack(
            weights, values, capacities, time_limit=60)
    else:
        print("Using greedy heuristic as reference")
        reference_x, reference_value = greedy_knapsack(
            weights, values, capacities)

    print(f"Reference value: {reference_value}")

    # sweep over Grover iteration counts (same role as QAOA p)
    results = []
    for k in [1, 2, 3, 4, 5, 6, 7]:
        print(f"Running QTG k={k}")
        qtg_x, qtg_value = run_knapsack(weights, values, capacities,
                                        sampler, k)
        results.append((k, qtg_value, reference_value))

    with open(output_file, "w") as f:
        for r in results:
            f.write(",".join(map(str, r)) + "\n")

    print(f"Results written to {output_file}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    output_file = sys.argv[1]
    n = int(sys.argv[2])

    print("in the file!", flush=True)

    run_p_tests(output_file, n)
