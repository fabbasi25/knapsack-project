# import libraries
from qaoa import run_knapsack 

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import matplotlib.pyplot as plt
import numpy as np 
from itertools import product


def brute_force_multiknapsack(weights, values, capacities):
    n = len(weights)  # number of items
    m = len(capacities)  # number of knapsacks

    best_value = 0
    best_x = None

    # x[i][j] = 1 if item j is in knapsack i
    for x in product([0,1], repeat=n*m):
        x = np.array(x).reshape(m, n)
        
        # each item can only be in one knapsack
        if np.any(x.sum(axis=0) > 1):
            continue
        
        # check capacity constraint for each knapsack
        if np.all(x @ weights <= capacities):
            val = np.sum(x * values)
            if val > best_value:
                best_value = val
                best_x = x

    return best_x, best_value

backend = AerSimulator()
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

sampler = AerSampler()


def parse_knapsack_file(filename):
    with open(filename, "r") as f:
        lines = f.read().strip().splitlines()

    n = int(lines[0])
    item_lines = lines[1:1+n]
    capacity_line = lines[1+n]

    weights = []
    values = []

    for line in item_lines:
        w, v = map(int, line.split())
        weights.append(w)
        values.append(v)

    capacity = [int(capacity_line)]

    return weights, values, capacity


all_data = []

def run_p_tests(input_file, output_file):
    w, v, c = parse_knapsack_file(input_file)
    correct, correct_value = brute_force_multiknapsack(w, v, c)

    for p in range(0, 51, 5):
        print(f"in the loop {p}")
        result, result_value = run_knapsack(w, v, c, sampler, p)
        print("calculated results")

        with open(output_file, "w") as o: 
            o.write(str(p) + "\n")
            o.write(str(result_value) + "\n")
            o.write(str(correct_value) + "\n")



import sys

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    run_p_tests(input_file, output_file)

