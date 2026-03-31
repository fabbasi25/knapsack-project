import math
import numpy as np
from numpy import pi

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.synthesis.qft import synth_qft_full
from qiskit_aer.primitives import Sampler as AerSampler

# ============================================================
# Core QTG Functions
# ============================================================

def solve_greedy(weights, profits, capacity):
    n = len(weights)
    ratios = [(profits[i] / weights[i], i) for i in range(n)]
    ratios.sort(reverse=True, key=lambda x: x[0])

    greedy_solution = [0] * n
    remaining_cap = capacity
    greedy_profit = 0

    for _, idx in ratios:
        if weights[idx] <= remaining_cap:
            greedy_solution[idx] = 1
            remaining_cap -= weights[idx]
            greedy_profit += profits[idx]

    return greedy_solution, greedy_profit


def compute_register_sizes(weights, profits, capacity):
    max_weight_sum = capacity
    n_cap = math.ceil(math.log2(max_weight_sum + 1)) + 1

    max_profit_sum = sum(profits)
    n_prof = math.ceil(math.log2(max_profit_sum + 1)) + 1

    return n_cap, n_prof


def ry_angle(greedy_bit, b):
    if greedy_bit == 0:
        return 2 * np.arccos(np.sqrt((1 + b) / (2 + b)))
    else:
        return 2 * np.arccos(np.sqrt(1 / (2 + b)))


_qft_cache = {}
_iqft_cache = {}

def qft_gate(n):
    if n not in _qft_cache:
        _qft_cache[n] = synth_qft_full(n, do_swaps=False).to_gate()
    return _qft_cache[n]

def iqft_gate(n):
    if n not in _iqft_cache:
        _iqft_cache[n] = qft_gate(n).inverse()
    return _iqft_cache[n]

def add_const_fourier(qc, reg, a):
    n = len(reg)
    for j in range(n):
        angle = 2 * pi * a / (2 ** (j + 1))
        qc.p(angle, reg[j])

def ctrl_add_const_fourier(qc, ctrl, reg, a):
    n = len(reg)
    for j in range(n):
        angle = 2 * pi * a / (2 ** (j + 1))
        qc.cp(angle, ctrl, reg[j])


def compare_geq(qc, cap_reg, ancilla, value, n_cap):
    msb = n_cap - 1
    qc.append(qft_gate(n_cap), cap_reg)
    add_const_fourier(qc, cap_reg, -value)
    qc.append(iqft_gate(n_cap), cap_reg)

    qc.x(cap_reg[msb])
    qc.cx(cap_reg[msb], ancilla)
    qc.x(cap_reg[msb])

    qc.append(qft_gate(n_cap), cap_reg)
    add_const_fourier(qc, cap_reg, value)
    qc.append(iqft_gate(n_cap), cap_reg)

uncompare_geq = compare_geq


def build_qtg(n_items, weights, profits, capacity, n_cap, n_prof, greedy_solution, b):
    total = n_items + n_cap + n_prof + 1
    qc = QuantumCircuit(total, name='QTG')

    path_q = list(range(0, n_items))
    cap_q = list(range(n_items, n_items + n_cap))
    prof_q = list(range(n_items + n_cap, n_items + n_cap + n_prof))
    anc_q = n_items + n_cap + n_prof

    for bit_idx in range(n_cap):
        if (capacity >> bit_idx) & 1:
            qc.x(cap_q[bit_idx])

    for m in range(n_items):
        theta = ry_angle(greedy_solution[m], b)
        
        compare_geq(qc, cap_q, anc_q, weights[m], n_cap)
        qc.cry(theta, anc_q, path_q[m])
        uncompare_geq(qc, cap_q, anc_q, weights[m], n_cap)

        qc.append(qft_gate(n_cap), cap_q)
        ctrl_add_const_fourier(qc, path_q[m], cap_q, -weights[m])
        qc.append(iqft_gate(n_cap), cap_q)

        qc.append(qft_gate(n_prof), prof_q)
        ctrl_add_const_fourier(qc, path_q[m], prof_q, profits[m])
        qc.append(iqft_gate(n_prof), prof_q)

    return qc

def threshold_oracle(qc, prof_q, anc_q, threshold, n_prof):
    msb = n_prof - 1

    qc.append(qft_gate(n_prof), prof_q)
    add_const_fourier(qc, prof_q, -threshold)
    qc.append(iqft_gate(n_prof), prof_q)

    qc.x(prof_q[msb])
    qc.cx(prof_q[msb], anc_q)
    qc.x(prof_q[msb])

    qc.z(anc_q)

    qc.x(prof_q[msb])
    qc.cx(prof_q[msb], anc_q)
    qc.x(prof_q[msb])

    qc.append(qft_gate(n_prof), prof_q)
    add_const_fourier(qc, prof_q, threshold)
    qc.append(iqft_gate(n_prof), prof_q)

def reflect_zero(qc, all_qubits):
    for q in all_qubits:
        qc.x(q)

    target = all_qubits[-1]
    controls = all_qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for q in all_qubits:
        qc.x(q)

def build_amplified_circuit(k_iters, qtg_circ, n_items, n_cap, n_prof, capacity, threshold):
    path_reg = QuantumRegister(n_items, 'path')
    cap_reg  = QuantumRegister(n_cap,   'cap')
    prof_reg = QuantumRegister(n_prof,  'prof')
    anc_reg  = QuantumRegister(1,       'anc')
    c_path   = ClassicalRegister(n_items, 'c_path')
    c_prof   = ClassicalRegister(n_prof,  'c_prof')

    qc = QuantumCircuit(path_reg, cap_reg, prof_reg, anc_reg, c_path, c_prof)

    all_q   = list(range(n_items + n_cap + n_prof + 1))
    prof_q  = list(range(n_items + n_cap, n_items + n_cap + n_prof))
    anc_idx = n_items + n_cap + n_prof

    qtg_gate = qtg_circ.to_gate(label='QTG')
    qtg_inv  = qtg_gate.inverse()
    qtg_inv.label = 'QTG_dag'

    qc.append(qtg_gate, all_q)

    for _ in range(k_iters):
        qc.barrier()
        threshold_oracle(qc, prof_q, anc_idx, threshold, n_prof)
        qc.barrier()
        qc.append(qtg_inv, all_q)
        qc.barrier()
        reflect_zero(qc, all_q)
        qc.barrier()
        qc.append(qtg_gate, all_q)

    qc.measure(path_reg, c_path)
    qc.measure(prof_reg, c_prof)

    return qc


# ============================================================
# run one amplification at a given threshold
# ============================================================

def _run_single_round(qtg_circ, n_items, n_cap, n_prof, capacity,
                      weights, profits, threshold, sampler,
                      k_iters, shots):
    """
    Execute QTG + k Grover iterations at the given threshold.

    Returns:
        best_profit:   profit of the best feasible state measured
        best_bitstr:   the corresponding path bitstring
        marked_states: list of (bitstr, profit) for all feasible states
                       whose profit >= threshold that appeared in samples
    """
    # Build and run the amplified circuit at this threshold
    qc = build_amplified_circuit(k_iters, qtg_circ, n_items, n_cap,
                                 n_prof, capacity, threshold)
    job = sampler.run([qc], shots=shots)
    counts = job.result().quasi_dists[0].binary_probabilities()

    # Parse measurement results into path-only probabilities
    path_probs = {}
    for bitstr, prob in counts.items():
        path_bits = bitstr[n_prof:]
        path_probs[path_bits] = path_probs.get(path_bits, 0) + prob

    # Scan every measured path: track the best feasible solution
    # and collect all "marked" states (feasible + profit >= threshold)
    best_profit = -1
    best_bitstr = None
    marked_set = set()

    for pb in path_probs:
        tw = sum(weights[i] for i in range(n_items) if pb[n_items-1-i] == '1')
        tp = sum(profits[i] for i in range(n_items) if pb[n_items-1-i] == '1')

        if tw <= capacity:
            # Track globally best feasible
            if tp > best_profit:
                best_profit = tp
                best_bitstr = pb
            # Track marked states (those the oracle would phase-flip)
            if tp >= threshold:
                marked_set.add((pb, tp))

    marked_states = sorted(marked_set, key=lambda x: -x[1])
    return best_profit, best_bitstr, marked_states


# ============================================================
# Iteration circuit to get the best solution
# ============================================================

def run_qtg_knapsack(weights, values, capacities, sampler, k_iters,
                     shots=1000, max_rounds=10):
    """
    Iterative QTG knapsack solver.

    Starting from threshold = greedy_profit + 1, the algorithm runs
    QTG + Grover amplification, then raises the threshold to the
    best profit found so far and repeats.  The loop stops when at
    most one feasible state is marked by the oracle, meaning we have
    isolated the single optimal solution.

    Args:
        weights (list):      Item weights.
        values (list):       Item values (profits).
        capacities (list):   List of capacities (uses the first one).
        sampler:             Qiskit AerSampler instance.
        k_iters (int):       Number of Grover iterations per round.
        shots (int):         Number of shots per circuit execution.
        max_rounds (int):    Safety cap on refinement iterations.

    Returns:
        best_x (np.array):           Shape (1, n_items) binary solution.
        best_measured_profit (int):   Total value of that solution.
    """
    n_items = len(weights)
    profits = values
    capacity = capacities[0]

    # ---- 1. Classical greedy preparation ----
    greedy_solution, greedy_profit = solve_greedy(weights, profits, capacity)
    n_cap, n_prof = compute_register_sizes(weights, profits, capacity)
    b = n_items / 4

    # ---- 2. Build the QTG circuit (reused every round) ----
    qtg_circ = build_qtg(n_items, weights, profits, capacity,
                         n_cap, n_prof, greedy_solution, b)

    # ---- 3. Iterative threshold refinement loop ----
    current_threshold = greedy_profit + 1

    # Keep track of the overall best across all rounds
    global_best_profit = greedy_profit
    global_best_bitstr = ''.join(
        str(greedy_solution[n_items-1-i]) for i in range(n_items))

    for round_num in range(1, max_rounds + 1):
        best_profit, best_bitstr, marked = _run_single_round(
            qtg_circ, n_items, n_cap, n_prof, capacity,
            weights, profits, current_threshold, sampler,
            k_iters, shots)

        # Update global best if this round found something better
        if best_profit > global_best_profit and best_bitstr is not None:
            global_best_profit = best_profit
            global_best_bitstr = best_bitstr

        # ----------------------------------------------------------
        # Convergence check
        # ----------------------------------------------------------
        # If 0 or 1 states are marked, we've isolated the optimum
        # (or overshot the threshold).  Either way, stop.
        if len(marked) <= 1:
            break

        # Multiple marked states still survive -> tighten the
        # threshold to the best profit we just observed, so only
        # the top-profit state(s) will be marked next round.
        new_threshold = best_profit
        if new_threshold <= current_threshold:
            # best_profit didn't exceed the current bar; nudge by 1
            # to guarantee the marked set shrinks.
            new_threshold = current_threshold + 1
        current_threshold = new_threshold

    # ---- 4. Format output for the benchmark harness ----
    # Fallback to greedy if no feasible solution was ever sampled
    if global_best_bitstr is None:
        global_best_profit = greedy_profit
        global_best_bitstr = ''.join(
            str(greedy_solution[n_items-1-i]) for i in range(n_items))

    best_x = np.zeros((1, n_items), dtype=int)
    for i in range(n_items):
        if global_best_bitstr[n_items-1-i] == '1':
            best_x[0, i] = 1

    return best_x, global_best_profit
