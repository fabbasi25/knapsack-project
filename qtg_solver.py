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
# Main Wrapper for Benchmark Script
# ============================================================

def run_qtg_knapsack(weights, values, capacities, sampler, k_iters, shots=1000):
    """
        
    Args:
        weights (list): Item weights.
        values (list): Item values (profits).
        capacities (list): List of capacities (extracts the first one).
        sampler: Qiskit AerSampler instance.
        k_iters (int): Number of Grover iterations (equivalent to 'p' depth).
        shots (int): Number of shots for the sampler.
        
    Returns:
        best_x (np.array): Matrix of shape (1, n_items) representing the bitstring.
        best_measured_profit (int): Total value of the best feasible solution found.
    """
    n_items = len(weights)
    profits = values
    capacity = capacities[0] # Assumes single knapsack from your generator

    # 1. Classical greedy preparation
    greedy_solution, greedy_profit = solve_greedy(weights, profits, capacity)
    n_cap, n_prof = compute_register_sizes(weights, profits, capacity)
    
    threshold = greedy_profit + 1
    b = n_items / 4

    # 2. Build Circuits
    qtg_circ = build_qtg(n_items, weights, profits, capacity, n_cap, n_prof, greedy_solution, b)
    qc = build_amplified_circuit(k_iters, qtg_circ, n_items, n_cap, n_prof, capacity, threshold)

    # 3. Execute
    job = sampler.run([qc], shots=shots)
    counts = job.result().quasi_dists[0].binary_probabilities()

    # 4. Parse output and extract highest feasible profit
    path_probs = {}
    for bitstr, prob in counts.items():
        # c_prof is tracked at the start of the string, path_bits at the end
        path_bits = bitstr[n_prof:] 
        path_probs[path_bits] = path_probs.get(path_bits, 0) + prob

    best_measured_profit = -1
    best_measured_bitstr = None

    for pb in path_probs:
        tw = sum(weights[i] for i in range(n_items) if pb[n_items-1-i] == '1')
        tp = sum(profits[i] for i in range(n_items) if pb[n_items-1-i] == '1')
        
        if tw <= capacity and tp > best_measured_profit:
            best_measured_profit = tp
            best_measured_bitstr = pb

    # Fallback to greedy if no feasible solution was found in the samples
    if best_measured_bitstr is None:
        best_measured_profit = greedy_profit
        best_measured_bitstr = ''.join(str(greedy_solution[n_items-1-i]) for i in range(n_items))

    # Format the bitstring into the numpy array format the benchmark expects
    best_x = np.zeros((1, n_items), dtype=int)
    for i in range(n_items):
        if best_measured_bitstr[n_items-1-i] == '1':
            best_x[0, i] = 1

    return best_x, best_measured_profit