# import libraries
import qiskit
from qiskit import QuantumCircuit,transpile

from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.circuit.library import QFT
from qiskit.circuit import Parameter

from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.visualization import plot_histogram, plot_bloch_multivector


import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import minimize


## C = 1, A, B = max(v_i,j)

def knapsack_hamiltonian(weights, values, capacities, gamma):
    weights = np.array(weights); values = np.array(values); capacities = np.array(capacities)

    maxv = 2 * np.max(values)
    n = len(weights) # number of items 
    m = len(capacities) # number of knapsacks 
    s = np.int64(np.floor(np.log2(capacities)) + 1) # number of slack bits 
    total_s = np.sum(s)

    qc = QuantumCircuit(n*m + total_s*m)

    for i in range(m): 
        b = np.array([_ for _ in range(s[i])])

        for j in range(n):
            # qubit i, j corresponds to i*n + j 
            # objective term 
            qc.rz(-gamma*values[i], i*n + j)

            # occupation term 1 
            qc.rz(maxv * gamma*(1-2*m), i*n + j)

            # all capacity terms 
            cap_linear_ij = -np.sum(weights*weights[j])/2 - np.sum(weights[j]*2**b[i])/2 + weights[j]*capacities[i]
            qc.rz(maxv * 2*cap_linear_ij*gamma, i*n + j)

            for jp in range(j+1, n): 
                if j != jp: 
                    qc.rzz(maxv * gamma*weights[j]*weights[jp]/2, i*n + j, i*n + jp)

            for ip in range(i+1, m): 
                # occupation term 2 
                if i != ip: 
                    qc.rzz(maxv * gamma/2, i*n + j, ip*n + j)

            for b_val in b: 
                qc.rzz(maxv * gamma*weights[j]*2**b_val, i*n + j, n*m + i*s[i] + b_val)
        
        for (i_b, b_val) in enumerate(b): 
            cap_linear_ib = -np.sum(2**(b_val+b[i]))/2 - np.sum(weights*2**b_val)/2 + 2**b_val*capacities[i]
            qc.rz(maxv * 2*cap_linear_ib*gamma, n*m + i*s[i] + b_val)

            for i_bp in range(i_b, len(b)):
                bp_val = b[i_bp]
                if b_val != bp_val: 
                    qc.rzz(maxv * 2**(b_val + bp_val)/2, n*m + i*s[i] + b_val, n*m + i*s[i] + bp_val)

    return qc 


def mixing_hamiltonian(N, beta): 
    qc = QuantumCircuit(N)
    
    for i in range(N): 
        qc.rx(2*beta, i)

    return qc 


def full_knapsack_circuit(weights, values, capacities, p): 
    n = len(weights) # number of items 
    m = len(capacities) # number of knapsacks 
    s = np.int64(np.floor(np.log2(capacities)) + 1) # number of slack bits 
    total_s = np.sum(s)
    N = n*m + total_s

    qc = QuantumCircuit(N, n*m)
    qc.h(range(N))

    for _ in range(p): 
        qc.compose(knapsack_hamiltonian(weights, values, capacities, Parameter(f"γ{_}")), inplace=True)
        qc.barrier()
        qc.compose(mixing_hamiltonian(N, Parameter(f"β{_}")), inplace=True)
        qc.barrier()

    qc.measure(range(n*m), range(n*m))
    return qc 


def objective(params, weights, values, capacities, p, sampler, shots=10000): 
    n = len(weights) # number of items 
    m = len(capacities) # number of knapsacks 

    qc = full_knapsack_circuit(weights, values, capacities, p)
    job = sampler.run([qc], [params], shots=shots)
    counts = job.result().quasi_dists[0].binary_probabilities()

    # compute expected value 
    expected = 0 
    for bitstring, prob in counts.items(): 
        # get all the items that were chosen 
        x = np.array([int(b) for b in reversed(bitstring)])

        assert len(x) == n*m, f"expected {n*m} bits, got {len(x)}"

        x = x.reshape(m, n)
        if all(np.dot(x[i], weights) <= capacities[i] for i in range(m)): 
            # i.e., if feasible 
            total_value = sum(x[i][j] * values[j] for i in range(m) for j in range(n))
            expected += prob * total_value
    
    return -expected # minimize -expected to maximize knapsack value 


def run_knapsack(weights, values, capacities, sampler, p=1):
    n = len(weights) # number of items 
    m = len(capacities) # number of knapsacks 

    best_value = 0
    best_bitstring = None
    for _ in range(5):
        result = minimize(objective, x0=np.random.uniform(0, np.pi, 2*p), args=(weights, values, capacities, p, sampler, 8192), method='COBYLA', options={'maxiter': 500})

        # now run with optimized params
        best_params = result.x

        mycirc = full_knapsack_circuit(weights, values, capacities, p)

        job_sim = sampler.run([mycirc], [best_params], shots=8192)

        quasi_dists = job_sim.result().quasi_dists[0].binary_probabilities()

        # check feasible out of most probable
        sorted_results = sorted(quasi_dists.items(), key=lambda x: x[1], reverse=True)
        soln = []
        soln_value = 0

        for bitstring, prob in sorted_results: 
            # get all the items that were chosen 
            x = np.array([int(b) for b in reversed(bitstring)])

            assert len(x) == n*m, f"expected {n*m} bits, got {len(x)}"

            x = x.reshape(m, n)
            if all(np.dot(x[i], weights) <= capacities[i] for i in range(m)): 
                soln = x
                soln_value = np.sum(np.array([np.dot(x[i], values) for i in range(m)]))
                break 
        
        if soln_value > best_value: 
            best_value = soln_value
            best_bitstring = soln 
        
    return best_bitstring, best_value

