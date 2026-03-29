import os
import sys
import time
import numpy as np
from itertools import product

from qtg_solver import run_qtg_knapsack
from qiskit_aer.primitives import Sampler as AerSampler

print("top of file", flush=True)

# Initialize the simulator once
sampler = AerSampler()

# ============================================================
# Classical Baselines
# ============================================================

def brute_force_multiknapsack(weights, values, capacities, time_limit=None):
    """Brute-force approach for small instances (n <= 20)."""
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
    """Greedy heuristic for large instances (n > 20)."""
    n = len(weights)
    m = len(capacities)
    best_x = np.zeros((m, n), dtype=int)
    remaining = capacities.copy()

    # Compute value/weight ratio
    ratio = [v / w for v, w in zip(values, weights)]
    order = np.argsort(ratio)[::-1]  # descending

    for j in order:
        # assign to the first knapsack that fits
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

def generate_knapsack(n, weight_range=(1, 50), value_range=(1, 100), capacity_ratio=0.5, seed=None):
    """Generate a random n-item single-knapsack instance."""
    rng = np.random.default_rng(seed)

    weights = rng.integers(*weight_range, size=n)
    values = rng.integers(*value_range, size=n)

    capacity = int(capacity_ratio * np.sum(weights))

    return weights.tolist(), values.tolist(), [capacity]

# ============================================================
# Main test function
# ============================================================

def run_p_tests(output_file, n):
    # Convert n to an integer just in case it came from sys.argv as a string
    n = int(n) 
    
    # Grab the Slurm array ID to use as a unique seed
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    weights, values, capacities = generate_knapsack(
        n=n,
        seed=task_id
    )

    print(f"Running task {task_id}", flush=True)

    if n <= 20:
        # small instances -> brute-force
        print("Running brute-force as reference")
        reference_x, reference_value = brute_force_multiknapsack(weights, values, capacities, time_limit=60)
    else:
        # large instances -> greedy heuristic
        print("Using greedy heuristic as reference")
        reference_x, reference_value = greedy_knapsack(weights, values, capacities)

    print(f"Reference value: {reference_value}")

    results = []
    
    for p in [0, 1, 2, 3, 4, 5]: # 'p' now acts as Grover iterations (k)
        print(f"Running QTG k={p}", flush=True)
        qtg_x, qtg_value = run_qtg_knapsack(weights, values, capacities, sampler, p)
        results.append((p, qtg_value, reference_value))

    # Write output to the specified file
    with open(output_file, "w") as f:
        for r in results:
            f.write(",".join(map(str, r)) + "\n")

    print(f"Results written to {output_file}", flush=True)

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    output_file = sys.argv[1]
    n = sys.argv[2]

    print("in the file!", flush=True)

    run_p_tests(output_file, n)