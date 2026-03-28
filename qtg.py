# ============================================================
# qtg.py — Quantum Tree Generator (QTG) for 0-1 Knapsack
# ============================================================
#
# Wraps the QTG algorithm from:
#   "A quantum algorithm for solving 0-1 Knapsack problems",
#   Wilkening et al., npj Quantum Information (2025)
#
# into a single callable function:
#
#   run_knapsack(weights, values, capacities, sampler, k_iters)
#
# This matches the same interface as the QAOA module used by
# tests_p.py, so you can swap:
#   from qaoa import run_knapsack   -->   from qtg import run_knapsack
#
# Parameters
# ----------
# weights    : list[int]   — item weights  (length n)
# values     : list[int]   — item profits  (length n)
# capacities : list[int]   — single-element list [capacity]
# sampler    : AerSampler  — Qiskit Aer sampler instance
# k_iters    : int         — number of Grover iterations (replaces QAOA's p)
#
# Returns
# -------
# best_x     : np.ndarray  — shape (1, n), binary assignment matrix
# best_value : int         — total profit of the best feasible solution found

import math
import numpy as np
from numpy import pi

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.synthesis.qft import synth_qft_full


# ============================================================
# Internal caches and low-level helpers  (from notebook cells 3-8)
# ============================================================

_qft_cache  = {}
_iqft_cache = {}


def _qft_gate(n):
    """Return a cached QFT gate for n qubits (no end-swaps)."""
    if n not in _qft_cache:
        _qft_cache[n] = synth_qft_full(n, do_swaps=False).to_gate()
    return _qft_cache[n]


def _iqft_gate(n):
    """Return a cached inverse-QFT gate for n qubits."""
    if n not in _iqft_cache:
        _iqft_cache[n] = _qft_gate(n).inverse()
    return _iqft_cache[n]


def _add_const_fourier(qc, reg, a):
    """Draper adder: add classical integer a to a register already
    in the Fourier basis.  Pass a negative value to subtract."""
    n = len(reg)
    for j in range(n):
        angle = 2 * pi * a / (2 ** (j + 1))
        qc.p(angle, reg[j])


def _ctrl_add_const_fourier(qc, ctrl, reg, a):
    """Controlled Draper adder: add a iff control qubit is |1>."""
    n = len(reg)
    for j in range(n):
        angle = 2 * pi * a / (2 ** (j + 1))
        qc.cp(angle, ctrl, reg[j])


# ============================================================
# Biased Ry angle  (Eq. 3 of the paper)
# ============================================================

def _ry_angle(greedy_bit, b):
    """Compute the biased Ry rotation angle for one item.

    If greedy took this item (greedy_bit=1), the angle favours
    inclusion.  If greedy skipped it (greedy_bit=0), the angle
    favours exclusion.  This biases the initial superposition
    towards the greedy neighbourhood, giving Grover a head start.
    """
    if greedy_bit == 0:
        return 2 * np.arccos(np.sqrt((1 + b) / (2 + b)))
    else:
        return 2 * np.arccos(np.sqrt(1 / (2 + b)))


# ============================================================
# Quantum comparator:  cap >= w?   (Eq. 4)
# ============================================================

def _compare_geq(qc, cap_reg, ancilla, value, n_cap):
    """Set ancilla = ancilla XOR (cap_reg >= value).

    Uses the subtract-then-check-MSB sign trick:
      1. QFT -> subtract value -> iQFT
      2. MSB=0 means non-negative (cap >= value)
      3. Copy sign into ancilla via X-CNOT-X
      4. Restore cap by adding value back
    """
    msb = n_cap - 1

    # subtract
    qc.append(_qft_gate(n_cap), cap_reg)
    _add_const_fourier(qc, cap_reg, -value)
    qc.append(_iqft_gate(n_cap), cap_reg)

    # read sign into ancilla
    qc.x(cap_reg[msb])
    qc.cx(cap_reg[msb], ancilla)
    qc.x(cap_reg[msb])

    # restore cap
    qc.append(_qft_gate(n_cap), cap_reg)
    _add_const_fourier(qc, cap_reg, value)
    qc.append(_iqft_gate(n_cap), cap_reg)


# Self-inverse: calling twice uncomputes the ancilla
_uncompare_geq = _compare_geq


# ============================================================
# Build the QTG circuit  (Box 1 / Eq. 4)
# ============================================================

def _build_qtg(n_items, weights, profits, capacity, n_cap, n_prof,
               greedy_solution, b):
    """Construct the full QTG unitary for a given knapsack instance.

    The QTG is a cascade  G = U_1 · U_2 · … · U_n.
    For each item m the layer U_m does:
      U^1_m : if cap >= w_m, apply biased Ry on path[m]
      U^2_m : if path[m]=1, subtract w_m from cap
      U^3_m : if path[m]=1, add p_m to profit

    Capacity is initialized inside the circuit so that
    QTG|0…0> = |ψ> and QTG†|ψ> = |0…0>  (needed for reflection).
    """
    total = n_items + n_cap + n_prof + 1
    qc = QuantumCircuit(total, name='QTG')

    # qubit index ranges
    path_q = list(range(0, n_items))
    cap_q  = list(range(n_items, n_items + n_cap))
    prof_q = list(range(n_items + n_cap, n_items + n_cap + n_prof))
    anc_q  = n_items + n_cap + n_prof

    # encode capacity in binary (LSB at index 0)
    for bit_idx in range(n_cap):
        if (capacity >> bit_idx) & 1:
            qc.x(cap_q[bit_idx])

    # process each item in density-sorted order
    for m in range(n_items):
        theta = _ry_angle(greedy_solution[m], b)

        # U^1_m: biased superposition conditioned on cap >= w_m
        _compare_geq(qc, cap_q, anc_q, weights[m], n_cap)
        qc.cry(theta, anc_q, path_q[m])
        _uncompare_geq(qc, cap_q, anc_q, weights[m], n_cap)

        # U^2_m: subtract w_m from cap, controlled on path[m]
        qc.append(_qft_gate(n_cap), cap_q)
        _ctrl_add_const_fourier(qc, path_q[m], cap_q, -weights[m])
        qc.append(_iqft_gate(n_cap), cap_q)

        # U^3_m: add p_m to profit, controlled on path[m]
        qc.append(_qft_gate(n_prof), prof_q)
        _ctrl_add_const_fourier(qc, path_q[m], prof_q, profits[m])
        qc.append(_iqft_gate(n_prof), prof_q)

    return qc


# ============================================================
# Threshold oracle  (phase-flip states with profit >= T)
# ============================================================

def _threshold_oracle(qc, prof_q, anc_q, threshold, n_prof):
    """Phase-flip every basis state whose accumulated profit >= threshold.

    Same subtract-check-MSB technique as the comparator, but applied
    to the profit register, followed by a Z gate on the ancilla to
    impart the −1 phase, then full uncomputation.
    """
    msb = n_prof - 1

    # compute ancilla = (profit >= threshold)
    qc.append(_qft_gate(n_prof), prof_q)
    _add_const_fourier(qc, prof_q, -threshold)
    qc.append(_iqft_gate(n_prof), prof_q)

    qc.x(prof_q[msb])
    qc.cx(prof_q[msb], anc_q)
    qc.x(prof_q[msb])

    # phase kick
    qc.z(anc_q)

    # uncompute ancilla
    qc.x(prof_q[msb])
    qc.cx(prof_q[msb], anc_q)
    qc.x(prof_q[msb])

    # restore profit register
    qc.append(_qft_gate(n_prof), prof_q)
    _add_const_fourier(qc, prof_q, threshold)
    qc.append(_iqft_gate(n_prof), prof_q)


# ============================================================
# Reflection about |0…0>   (S_0 = 2|0><0| − I)
# ============================================================

def _reflect_zero(qc, all_qubits):
    """Reflect about |0…0> on the given qubits.

    X-all → multi-controlled-Z on |1…1> → X-all.
    The global phase of −1 is physically unobservable.
    """
    for q in all_qubits:
        qc.x(q)

    target   = all_qubits[-1]
    controls = all_qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for q in all_qubits:
        qc.x(q)


# ============================================================
# Classical preprocessing: sort + greedy + register sizing
# ============================================================

def _preprocess(weights, values, capacity):
    """Sort items by decreasing density, run greedy, compute
    register sizes.  Returns all parameters the quantum circuit needs.

    This is the classical preprocessing step prescribed by the paper:
    items are sorted by profit/weight ratio so the greedy algorithm
    (and hence the QTG bias) works in the best possible order.
    """
    n = len(weights)

    # sort items by decreasing profit-to-weight density
    items = list(zip(weights, values, range(n)))
    items.sort(key=lambda x: x[1] / x[0], reverse=True)

    sorted_weights = [x[0] for x in items]
    sorted_profits = [x[1] for x in items]
    # keep track of original indices so we can map back later
    original_idx   = [x[2] for x in items]

    # greedy: walk sorted list, take each item if it fits
    greedy_sol = [0] * n
    remaining  = capacity
    greedy_profit = 0
    for i in range(n):
        if sorted_weights[i] <= remaining:
            greedy_sol[i] = 1
            remaining -= sorted_weights[i]
            greedy_profit += sorted_profits[i]

    threshold = greedy_profit + 1  # search for strictly better

    # capacity register: 2^(n_cap-1) must exceed both capacity
    # and max weight, to avoid wrap-around in the sign trick
    n_cap = 1
    while 2 ** (n_cap - 1) <= capacity or 2 ** (n_cap - 1) <= max(sorted_weights):
        n_cap += 1

    # profit register: 2^(n_prof-1) must exceed threshold
    # (worst case: profit=0, subtract T → must stay representable)
    n_prof = 1
    while 2 ** (n_prof - 1) <= threshold:
        n_prof += 1

    # bias parameter  b = n / Δ  with Δ = 4  (paper's default)
    b = n / 4

    return {
        'n_items':         n,
        'weights':         sorted_weights,
        'profits':         sorted_profits,
        'original_idx':    original_idx,
        'capacity':        capacity,
        'greedy_solution': greedy_sol,
        'greedy_profit':   greedy_profit,
        'threshold':       threshold,
        'n_cap':           n_cap,
        'n_prof':          n_prof,
        'b':               b,
    }


# ============================================================
# Build the full amplified circuit
# ============================================================

def _build_amplified_circuit(k_iters, qtg_circ, n_items, n_cap,
                             n_prof, threshold):
    """Assemble the measurement-ready circuit:

        QTG → [ Oracle → QTG† → S_0 → QTG ] × k → Measure

    k_iters=0 gives the bare QTG with no amplification.
    """
    path_reg = QuantumRegister(n_items, 'path')
    cap_reg  = QuantumRegister(n_cap,   'cap')
    prof_reg = QuantumRegister(n_prof,  'prof')
    anc_reg  = QuantumRegister(1,       'anc')
    c_path   = ClassicalRegister(n_items, 'c_path')
    c_prof   = ClassicalRegister(n_prof,  'c_prof')

    qc = QuantumCircuit(path_reg, cap_reg, prof_reg, anc_reg,
                        c_path, c_prof)

    all_q   = list(range(n_items + n_cap + n_prof + 1))
    prof_q  = list(range(n_items + n_cap, n_items + n_cap + n_prof))
    anc_idx = n_items + n_cap + n_prof

    qtg_gate = qtg_circ.to_gate(label='QTG')
    qtg_inv  = qtg_gate.inverse()
    qtg_inv.label = 'QTG_dag'

    # initial state preparation
    qc.append(qtg_gate, all_q)

    # Grover iterations
    for _ in range(k_iters):
        _threshold_oracle(qc, prof_q, anc_idx, threshold, n_prof)
        qc.append(qtg_inv, all_q)
        _reflect_zero(qc, all_q)
        qc.append(qtg_gate, all_q)

    # measure path and profit registers
    qc.measure(path_reg, c_path)
    qc.measure(prof_reg, c_prof)

    return qc


# ============================================================
# Public API  —  matches the signature used by tests_p.py
# ============================================================

def run_knapsack(weights, values, capacities, sampler, k_iters,
                 shots=8192):
    """Solve a single 0-1 knapsack instance with the QTG algorithm.

    Parameters
    ----------
    weights    : list[int]  — item weights (unsorted, length n)
    values     : list[int]  — item profits / values (length n)
    capacities : list[int]  — **single-element** list [capacity]
    sampler    : AerSampler — Qiskit Aer Sampler instance
    k_iters    : int        — number of Grover iterations
                              (analogous to QAOA depth p)
    shots      : int        — number of measurement shots (default 8192)

    Returns
    -------
    best_x     : np.ndarray — shape (1, n), the best feasible binary
                              assignment found, in **original** item order
    best_value : int        — total profit of that assignment
    """
    capacity = capacities[0]
    n = len(weights)

    # ----------------------------------------------------------
    # 1. Classical preprocessing: sort, greedy, register sizing
    # ----------------------------------------------------------
    pp = _preprocess(weights, values, capacity)

    # ----------------------------------------------------------
    # 2. Build QTG circuit for this instance
    # ----------------------------------------------------------
    qtg_circ = _build_qtg(
        pp['n_items'], pp['weights'], pp['profits'], pp['capacity'],
        pp['n_cap'], pp['n_prof'], pp['greedy_solution'], pp['b'],
    )

    # ----------------------------------------------------------
    # 3. Build the amplified circuit with k Grover iterations
    # ----------------------------------------------------------
    qc = _build_amplified_circuit(
        k_iters, qtg_circ,
        pp['n_items'], pp['n_cap'], pp['n_prof'], pp['threshold'],
    )

    # ----------------------------------------------------------
    # 4. Run on the simulator
    # ----------------------------------------------------------
    job    = sampler.run([qc], shots=shots)
    counts = job.result().quasi_dists[0].binary_probabilities()

    # ----------------------------------------------------------
    # 5. Parse measurement results and find best feasible solution
    # ----------------------------------------------------------
    n_items = pp['n_items']
    n_prof  = pp['n_prof']
    sorted_weights = pp['weights']
    sorted_profits = pp['profits']
    original_idx   = pp['original_idx']

    best_value = 0
    best_sorted_bits = [0] * n_items   # in sorted order

    for bitstr, prob in counts.items():
        # bitstring layout: [c_prof bits] [c_path bits]
        # both are MSB-first; path bit i (left to right) = item (n_items-1-i)
        path_bits = bitstr[n_prof:]

        # decode which items (in sorted order) are selected
        tw = 0
        tp = 0
        bits = []
        for i in range(n_items):
            # path_bits[n_items-1-i] is the measurement bit for sorted item i
            selected = int(path_bits[n_items - 1 - i])
            bits.append(selected)
            if selected:
                tw += sorted_weights[i]
                tp += sorted_profits[i]

        # keep only feasible solutions, pick the best profit
        if tw <= capacity and tp > best_value:
            best_value = tp
            best_sorted_bits = bits

    # ----------------------------------------------------------
    # 6. Map back to original item ordering
    # ----------------------------------------------------------
    # best_sorted_bits[i] tells whether sorted-item i was selected.
    # original_idx[i] is the original position of sorted-item i.
    best_x = np.zeros((1, n), dtype=int)
    for i in range(n_items):
        best_x[0, original_idx[i]] = best_sorted_bits[i]

    return best_x, best_value
