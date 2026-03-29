# import libraries
import time
import numpy as np
from itertools import product

print("top of file", flush=True)

from qaoa10 import run_knapsack
from qiskit_aer.primitives import Sampler as AerSampler

sampler = AerSampler()

# -------------------------
# Brute-force with time limit
# -------------------------
def brute_force_multiknapsack(weights, values, capacities, time_limit=None):
    n = len(weights)
    m = len(capacities)
    best_value = 0
    best_x = None
    start_time = time.time()

    for x in product([0,1], repeat=n*m):
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

# -------------------------
# Greedy heuristic for large instances
# -------------------------
def greedy_knapsack(weights, values, capacities):
    n = len(weights)
    m = len(capacities)
    best_x = np.zeros((m,n), dtype=int)
    remaining = capacities.copy()

    # Compute value/weight ratio
    ratio = [v/w for v, w in zip(values, weights)]
    order = np.argsort(ratio)[::-1]  # descending

    for j in order:
        # assign to the first knapsack that fits
        for i in range(m):
            if weights[j] <= remaining[i]:
                best_x[i,j] = 1
                remaining[i] -= weights[j]
                break
    total_value = np.sum(best_x * values)
    return best_x, total_value

# -------------------------
# Main test function
# -------------------------

import os

import numpy as np

def generate_knapsack(n, weight_range=(1, 50), value_range=(1, 100), capacity_ratio=0.5, seed=None):
    rng = np.random.default_rng(seed)

    weights = rng.integers(*weight_range, size=n)
    values = rng.integers(*value_range, size=n)

    capacity = int(capacity_ratio * np.sum(weights))

    return weights.tolist(), values.tolist(), [capacity]


def run_p_tests(output_file, n):
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    for i in range(50):
        weights, values, capacities = generate_knapsack(
            n=n,
            seed=task_id + i
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
        for p in [1, 2, 3, 4, 5, 6, 7]:
            print(f"Running QAOA p={p}")
            qaoa_x, qaoa_value = run_knapsack(weights, values, capacities, sampler, p)
            results.append((p, qaoa_value, reference_value))

        with open(output_file, "a") as f:
            for r in results:
                f.write(",".join(map(str, r)) + "\n")

        print(f"Results written to {output_file}")


# def run_p_tests(input_file, output_file):
#     # parse instance
#     with open(input_file, "r") as f:
#         lines = f.read().strip().splitlines()
#     n = int(lines[0])
#     item_lines = lines[1:1+n]
#     capacity_line = lines[1+n]

#     weights = []
#     values = []
#     for line in item_lines:
#         w, v = map(int, line.split())
#         weights.append(w)
#         values.append(v)

#     capacities = [int(capacity_line)]

#     print(f"Processing {input_file} with {n} items")

#     # -------------------------
#     # Choose classical reference
#     # -------------------------
#     if n <= 20:
#         # small instances -> brute-force
#         print("Running brute-force as reference")
#         reference_x, reference_value = brute_force_multiknapsack(weights, values, capacities, time_limit=60)
#     else:
#         # large instances -> greedy heuristic
#         print("Using greedy heuristic as reference")
#         reference_x, reference_value = greedy_knapsack(weights, values, capacities)

#     print(f"Reference value: {reference_value}")

#     # -------------------------
#     # Run QAOA for different p
#     # -------------------------
#     results = []
#     for p in range(1, 51, 5):
#         print(f"Running QAOA p={p}")
#         qaoa_x, qaoa_value = run_knapsack(weights, values, capacities, sampler, p)

#         # compute "success probability" relative to reference
#         success = 1.0 if qaoa_value >= reference_value else 0.0
#         print(f"QAOA value: {qaoa_value}, success={success}")

#         results.append((p, qaoa_value, reference_value, success))

#     # -------------------------
#     # Write results
#     # -------------------------
#     with open(output_file, "w") as o:
#         o.write("p,qaoa_value,reference_value,success\n")
#         for r in results:
#             o.write(",".join(map(str,r)) + "\n")

#     print(f"Results written to {output_file}")


sampler = AerSampler()

# def parse_knapsack_file(filename):
#     with open(filename, "r") as f:
#         lines = f.read().strip().splitlines()

#     n = int(lines[0])
#     item_lines = lines[1:1+n]
#     capacity_line = lines[1+n]

#     weights = []
#     values = []

#     for line in item_lines:
#         w, v = map(int, line.split())
#         weights.append(w)
#         values.append(v)

#     capacity = [int(capacity_line)]

#     return weights, values, capacity


# all_data = []

# def run_p_tests(input_file, output_file):
#     w, v, c = parse_knapsack_file(input_file)
#     print("computing bruteforce!")
#     correct, correct_value = brute_force_multiknapsack(w, v, c)

#     for p in range(0, 51, 5):
#         print(f"in the loop {p}")
#         result, result_value = run_knapsack(w, v, c, sampler, p)
#         print("calculated results")

#         with open(output_file, "w") as o: 
#             o.write(str(p) + "\n")
#             o.write(str(result_value) + "\n")
#             o.write(str(correct_value) + "\n")


import sys

if __name__ == "__main__":
    output_file = sys.argv[1]
    n = int(sys.argv[2])

    run_p_tests(output_file, n)

