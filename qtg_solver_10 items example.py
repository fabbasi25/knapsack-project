import time
import random
import math
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.synthesis.qft import synth_qft_full
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.visualization import plot_histogram

# ============================================================
# Cell 1 -- Greedy solver ( computes greedy solution)
# ============================================================

def solve_greedy(weights, profits, capacity):
    """
    Classic greedy heuristic for 0/1 knapsack:
    Sort items by profit-to-weight ratio (descending),
    then greedily include items that still fit.
    
    Returns:
        greedy_solution: list of 0/1 for each item (in ORIGINAL order)
        greedy_profit:   total profit of the greedy solution
    """
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


# ============================================================
# Cell 2 -- Register size calculator
# ============================================================

def compute_register_sizes(weights, profits, capacity):
    max_weight_sum = capacity
    n_cap = math.ceil(math.log2(max_weight_sum + 1)) + 1

    max_profit_sum = sum(profits)
    n_prof = math.ceil(math.log2(max_profit_sum + 1)) + 1

    return n_cap, n_prof


# ============================================================
# Cell 3 -- Problem setup  (10-item instance example)
# ============================================================

weights = [2, 3, 4, 5, 3, 6, 7, 4, 8, 5]
profits = [6, 8, 10, 11, 6, 10, 11, 6, 10, 5]
capacity = 20

n_items = len(weights)
assert len(profits) == n_items, "weights and profits must have the same length"

greedy_solution, greedy_profit = solve_greedy(weights, profits, capacity)
n_cap, n_prof = compute_register_sizes(weights, profits, capacity)

# Initial threshold: we want strictly better than greedy
threshold = greedy_profit + 1
b = n_items / 4

print(f'Items ({n_items}):  weights={weights}, profits={profits}')
print(f'Capacity: {capacity}')
print(f'Greedy: {greedy_solution}  profit={greedy_profit}')
print(f'Initial threshold: profit >= {threshold}')
print(f'Bias b = {b}')
print(f'Registers: n_cap={n_cap}, n_prof={n_prof}')
print(f'Total qubits: {n_items}+{n_cap}+{n_prof}+1 = {n_items+n_cap+n_prof+1}')

# ============================================================
# Cell 4 -- Biased Ry angles
# ============================================================

def ry_angle(greedy_bit, b):
    if greedy_bit == 0:
        return 2 * np.arccos(np.sqrt((1 + b) / (2 + b)))
    else:
        return 2 * np.arccos(np.sqrt(1 / (2 + b)))

for m in range(n_items):
    theta = ry_angle(greedy_solution[m], b)
    p_inc = np.sin(theta / 2) ** 2
    print(f'Item {m}: w={weights[m]}, p={profits[m]}, '
          f'greedy={greedy_solution[m]}, '
          f'theta={theta:.4f}, '
          f'P(incl)={p_inc:.4f}, P(excl)={1-p_inc:.4f}')

# ============================================================
# Cell 5 -- QFT gate helpers and Draper adders
# ============================================================

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

print('QFT helpers and Draper adders defined.')

# ============================================================
# Cell 6 -- Quantum comparator: cap >= w?
# ============================================================

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
print('Quantum comparator defined.')

# ============================================================
# Cell 7 -- Build the QTG circuit
# ============================================================

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

qtg_circ = build_qtg(n_items, weights, profits, capacity, n_cap, n_prof, greedy_solution, b)
print(f'QTG built: {qtg_circ.num_qubits} qubits (path:{n_items} + cap:{n_cap} + prof:{n_prof} + anc:1)')

# ============================================================
# Cell 8 -- Threshold oracle
# ============================================================

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

print(f'Threshold oracle defined.')

# ============================================================
# Cell 9 -- Reflection about |0...0>
# ============================================================

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

print('Reflection about |0> defined.')

# ============================================================
# Cell 10 -- Run bare QTG once to see the initial distribution
# ============================================================

path_reg = QuantumRegister(n_items, 'path')
cap_reg  = QuantumRegister(n_cap, 'cap')
prof_reg = QuantumRegister(n_prof, 'prof')
anc_reg  = QuantumRegister(1, 'anc')
c_path   = ClassicalRegister(n_items, 'c_path')
c_prof   = ClassicalRegister(n_prof, 'c_prof')

qc_test = QuantumCircuit(path_reg, cap_reg, prof_reg, anc_reg, c_path, c_prof)

all_qubits = list(range(qc_test.num_qubits))
qc_test.append(qtg_circ.to_gate(), all_qubits)
qc_test.measure(path_reg, c_path)
qc_test.measure(prof_reg, c_prof)

sampler = AerSampler()
job = sampler.run([qc_test], shots=5000)
counts = job.result().quasi_dists[0].binary_probabilities()

path_probs = {}
for bitstr, prob in counts.items():
    path_bits = bitstr[n_prof:]
    path_probs[path_bits] = path_probs.get(path_bits, 0) + prob

greedy_bitstr = ''.join(str(greedy_solution[n_items-1-i]) for i in range(n_items))

print('QTG output (no amplification):')
print(f'{"Path":>12} {"Weight":>6} {"Profit":>6} {"Feasible":>8} {"Prob":>8}')
print('-' * 52)
for pb in sorted(path_probs.keys(), key=lambda x: -path_probs[x])[:20]:
    tw = sum(weights[i] for i in range(n_items) if pb[n_items-1-i]=='1')
    tp = sum(profits[i] for i in range(n_items) if pb[n_items-1-i]=='1')
    f = tw <= capacity
    tag = ''
    if pb == greedy_bitstr: tag = ' <-- greedy'
    print(f'{pb:>12} {tw:>6} {tp:>6} {str(f):>8} {path_probs[pb]:>8.4f}{tag}')

# ============================================================
# Cell 11 -- Histogram of bare QTG
# ============================================================

def color_bar(label, weights, profits, n_items, capacity, threshold):
    tw = sum(weights[i] for i in range(n_items) if label[n_items-1-i]=='1')
    tp = sum(profits[i] for i in range(n_items) if label[n_items-1-i]=='1')
    if tp >= threshold and tw <= capacity: return 'green'
    elif tw <= capacity:                   return 'blue'
    else:                                  return 'red'

labels = sorted(path_probs.keys(), key=lambda x: -path_probs[x])[:30]
colors = [color_bar(l, weights, profits, n_items, capacity, threshold)
          for l in labels]

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(range(len(labels)), [path_probs.get(l,0) for l in labels],
       color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_xlabel('Path (item selection)')
ax.set_ylabel('Probability')
ax.set_title(f'QTG Output -- {n_items} Items, No Amplification (top 30 paths)')
ax.legend(handles=[
    Patch(facecolor='green', edgecolor='black', label=f'Optimal (profit>={threshold})'),
    Patch(facecolor='blue', edgecolor='black', label=f'Feasible (weight<={capacity})'),
    Patch(facecolor='red', edgecolor='black', label='Infeasible'),
], loc='upper right')
plt.tight_layout()
plt.savefig('bare_qtg.png', dpi=150)
plt.show()

# ============================================================
# Cell 12 -- Build amplified circuit with k Grover iterations
# ============================================================

def build_amplified_circuit(k_iters, qtg_circ, n_items, n_cap,
                            n_prof, capacity, threshold):
    """Return a measurement-ready circuit with k Grover iterations
    using the given threshold for the oracle."""

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

    # Step 1: initial QTG
    qc.append(qtg_gate, all_q)

    # Steps 2-5: repeat k times
    for _ in range(k_iters):
        qc.barrier()
        threshold_oracle(qc, prof_q, anc_idx, threshold, n_prof)
        qc.barrier()
        qc.append(qtg_inv, all_q)
        qc.barrier()
        reflect_zero(qc, all_q)
        qc.barrier()
        qc.append(qtg_gate, all_q)

    # Measure
    qc.measure(path_reg, c_path)
    qc.measure(prof_reg, c_prof)

    return qc

print('build_amplified_circuit() defined.')

# ============================================================
# Cell 13 -- run one round of amplification at a given
#            threshold, sweep k=0..max_iters, return the best
#            feasible solution found and all probabilities.
# ============================================================

def run_amplification_round(threshold, qtg_circ, n_items, n_cap, n_prof,
                            capacity, weights, profits, max_iters=5,
                            shots=5000):
    """
    For a given threshold, sweep over Grover iterations k=0..max_iters.
    
    Returns:
        best_profit:    profit of the best feasible state found
        best_bitstr:    the bitstring of that state
        best_k:         the iteration count that maximised its probability
        all_path_probs: dict  k -> {path_bitstring: probability}
        marked_states:  list of (bitstr, profit) for all states with
                        profit >= threshold and weight <= capacity
    """
    sampler = AerSampler()
    all_path_probs = {}

    for k in range(max_iters + 1):
        print(f'  k={k} ...', end=' ', flush=True)
        qc_k = build_amplified_circuit(k, qtg_circ, n_items, n_cap,
                                       n_prof, capacity, threshold)
        job_k = sampler.run([qc_k], shots=shots)
        counts_k = job_k.result().quasi_dists[0].binary_probabilities()

        pp = {}
        for bitstr, prob in counts_k.items():
            path_bits = bitstr[n_prof:]
            pp[path_bits] = pp.get(path_bits, 0) + prob

        all_path_probs[k] = pp
        print('done')

    # ----------------------------------------------------------
    # Across all k values, find the single best feasible solution
    # and also enumerate all "marked" states (profit >= threshold).
    # ----------------------------------------------------------
    best_profit = -1
    best_bitstr = None
    marked_set = set()

    for k, pp in all_path_probs.items():
        for pb in pp:
            tw = sum(weights[i] for i in range(n_items) if pb[n_items-1-i] == '1')
            tp = sum(profits[i] for i in range(n_items) if pb[n_items-1-i] == '1')
            if tw <= capacity and tp >= threshold:
                marked_set.add((pb, tp))
            if tw <= capacity and tp > best_profit:
                best_profit = tp
                best_bitstr = pb

    # Find k that maximises the best state's probability
    best_k = max(range(max_iters + 1),
                 key=lambda k: all_path_probs[k].get(best_bitstr, 0)
                 if best_bitstr else 0)

    marked_states = sorted(marked_set, key=lambda x: -x[1])

    return best_profit, best_bitstr, best_k, all_path_probs, marked_states


# ============================================================
# Cell 14 -- ITERATIVE THRESHOLD REFINEMENT LOOP
#
#   The core idea:
#     1. Start with threshold = greedy_profit + 1
#     2. Run QTG + Grover sweeps at that threshold
#     3. If multiple states are marked (profit >= threshold),
#        raise the threshold to (best_profit_found) so that
#        only the true optimum survives the oracle.
#     4. Repeat until exactly one state is marked, or the
#        threshold can no longer be raised.
# ============================================================

MAX_OUTER_ROUNDS = 10        # safety cap on iterations
GROVER_MAX_K     = 5         # max Grover iterations per round
SHOTS            = 5000      # shots per circuit

current_threshold = threshold   # start from greedy + 1

# Store history for plotting
round_history = []              # list of dicts with round info


for round_num in range(1, MAX_OUTER_ROUNDS + 1):
    print(f'\n--- Round {round_num}  |  threshold = {current_threshold} ---')

    best_profit, best_bitstr, best_k, all_pp, marked = \
        run_amplification_round(
            current_threshold, qtg_circ, n_items, n_cap, n_prof,
            capacity, weights, profits,
            max_iters=GROVER_MAX_K, shots=SHOTS)

    # Decode the best solution for display
    if best_bitstr:
        best_items = [i for i in range(n_items) if best_bitstr[n_items-1-i] == '1']
        best_weight = sum(weights[i] for i in best_items)
    else:
        best_items = []
        best_weight = 0

    print(f'\n  Marked states (profit >= {current_threshold}):')
    for bs, pr in marked:
        items = [i for i in range(n_items) if bs[n_items-1-i] == '1']
        w = sum(weights[i] for i in items)
        tag = ' <-- BEST' if bs == best_bitstr else ''
        print(f'    {bs}  profit={pr}  weight={w}  items={items}{tag}')

    print(f'  Best found: profit={best_profit}, weight={best_weight}, '
          f'items={best_items}, best_k={best_k}')

    # Save this round's data
    round_history.append({
        'round': round_num,
        'threshold': current_threshold,
        'best_profit': best_profit,
        'best_bitstr': best_bitstr,
        'best_k': best_k,
        'marked_states': marked,
        'all_path_probs': all_pp,
    })

    # ----------------------------------------------------------
    # Convergence check:
    #   - If only 1 state is marked, we've isolated the optimum!
    #   - If 0 states are marked, the threshold was too high;
    #     we revert to the previous round's best.
    #   - Otherwise, raise the threshold and repeat.
    # ----------------------------------------------------------
    if len(marked) <= 1:
        print(f'\n  >>> CONVERGED: only {len(marked)} state(s) marked. Done!')
        break

    # Multiple marked states -> raise threshold to best_profit
    # so only states with profit >= best_profit survive.
    # (If best_profit == current_threshold, bump by 1 to shrink
    #  the marked set.)
    new_threshold = best_profit
    if new_threshold <= current_threshold:
        new_threshold = current_threshold + 1

    print(f'\n  Multiple marked states ({len(marked)}). '
          f'Raising threshold: {current_threshold} -> {new_threshold}')
    current_threshold = new_threshold

else:
    print(f'\n  Reached max rounds ({MAX_OUTER_ROUNDS}) without '
          f'isolating a single state.')

# ============================================================
# Cell 15 -- Final result summary
# ============================================================

final = round_history[-1]
print('\n' + '=' * 60)
print('  FINAL RESULT')
print('=' * 60)
print(f'  Greedy profit:   {greedy_profit}   solution: {greedy_solution}')
print(f'  Optimal profit:  {final["best_profit"]}')
if final['best_bitstr']:
    opt_items = [i for i in range(n_items)
                 if final['best_bitstr'][n_items-1-i] == '1']
    opt_weight = sum(weights[i] for i in opt_items)
    opt_solution = [1 if i in opt_items else 0 for i in range(n_items)]
    print(f'  Optimal items:   {opt_items}')
    print(f'  Optimal weight:  {opt_weight}  (capacity={capacity})')
    print(f'  Optimal vector:  {opt_solution}')
    print(f'  Final threshold: {final["threshold"]}')
    print(f'  Rounds needed:   {len(round_history)}')
    print(f'  Best Grover k:   {final["best_k"]}')
print('=' * 60)


# ============================================================
# Cell 16 -- Plot: convergence across rounds
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: threshold and marked-state count per round
rounds = [r['round'] for r in round_history]
thresholds = [r['threshold'] for r in round_history]
n_marked = [len(r['marked_states']) for r in round_history]

ax1 = axes[0]
ax1.plot(rounds, thresholds, 'ro-', markersize=8, linewidth=2, label='Threshold')
ax1.set_xlabel('Round', fontsize=12)
ax1.set_ylabel('Threshold', fontsize=12, color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_xticks(rounds)

ax1b = ax1.twinx()
ax1b.bar(rounds, n_marked, alpha=0.3, color='blue', label='# Marked states')
ax1b.set_ylabel('# Marked states', fontsize=12, color='blue')
ax1b.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Threshold Refinement Convergence', fontsize=13)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

# Right: final round side-by-side (k=0 vs best_k)
final_pp = final['all_path_probs']
final_threshold = final['threshold']
fk = final['best_k']

all_labels_set = set()
for pp in final_pp.values():
    all_labels_set.update(pp.keys())
all_labels = sorted(all_labels_set,
                    key=lambda l: max(final_pp[k].get(l, 0)
                                      for k in final_pp),
                    reverse=True)[:30]

colors_final = [color_bar(l, weights, profits, n_items, capacity,
                           final_threshold) for l in all_labels]

ax2 = axes[1]
pb_vals = [final_pp[fk].get(l, 0) for l in all_labels]
ax2.bar(range(len(all_labels)), pb_vals, color=colors_final,
        edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(all_labels)))
ax2.set_xticklabels(all_labels, rotation=90, fontsize=7)
ax2.set_title(f'Final Round: threshold={final_threshold}, k={fk}', fontsize=13)
ax2.set_xlabel('Path')
ax2.set_ylabel('Probability')
ax2.legend(handles=[
    Patch(facecolor='green', edgecolor='black',
          label=f'Optimal (profit>={final_threshold})'),
    Patch(facecolor='blue', edgecolor='black',
          label=f'Feasible (weight<={capacity})'),
    Patch(facecolor='red', edgecolor='black', label='Infeasible'),
], loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('iterative_convergence.png', dpi=150)
plt.show()


# ============================================================
# Cell 17 -- Per-round histograms showing how the marked set
#            shrinks at each round
# ============================================================

n_rounds = len(round_history)
fig, axes = plt.subplots(1, n_rounds, figsize=(8 * n_rounds, 5),
                         sharey=True, squeeze=False)

for idx, rinfo in enumerate(round_history):
    ax = axes[0][idx]
    rpp = rinfo['all_path_probs']
    rk = rinfo['best_k']
    rt = rinfo['threshold']

    # Collect all paths seen in this round
    round_labels_set = set()
    for pp in rpp.values():
        round_labels_set.update(pp.keys())
    round_labels = sorted(round_labels_set,
                          key=lambda l: max(rpp[k].get(l, 0)
                                            for k in rpp),
                          reverse=True)[:30]

    rcolors = [color_bar(l, weights, profits, n_items, capacity, rt)
               for l in round_labels]
    rvals = [rpp[rk].get(l, 0) for l in round_labels]

    ax.bar(range(len(round_labels)), rvals, color=rcolors,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(round_labels)))
    ax.set_xticklabels(round_labels, rotation=90, fontsize=7)
    ax.set_title(f'Round {rinfo["round"]}: thresh={rt}, k={rk}\n'
                 f'Marked: {len(rinfo["marked_states"])}', fontsize=11)
    ax.set_xlabel('Path')
    if idx == 0:
        ax.set_ylabel('Probability')

fig.legend(handles=[
    Patch(facecolor='green', edgecolor='black', label='Marked (above threshold)'),
    Patch(facecolor='blue', edgecolor='black', label='Feasible (below threshold)'),
    Patch(facecolor='red', edgecolor='black', label='Infeasible'),
], loc='upper center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.savefig('per_round_histograms.png', dpi=150)
plt.show()

print('\nAll plots saved.')
