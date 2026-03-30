import os
import glob
import collections
import matplotlib.pyplot as plt

# ── 1. Parse all files ───────────────────────────────────────────────────────
# data[p][n] = list of accuracy values (to be averaged)
data = collections.defaultdict(lambda: collections.defaultdict(list))

for filepath in glob.glob("knapsack-project/solutions/small_10_8192/*-f.txt"):
    filename = os.path.basename(filepath)
    n = int(filename.split("-")[0])

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue
            p, quantum, classical = int(parts[0]), float(parts[1]), float(parts[2])
            accuracy = quantum / classical if classical != 0 else 0.0
            data[p][n].append(accuracy)

# ── 2. Average and sort ───────────────────────────────────────────────────────
averaged = {}  # averaged[p] = [(n, mean_accuracy), ...]
for p in data:
    averaged[p] = sorted(
        [(n, sum(vals) / len(vals)) for n, vals in data[p].items()],
        key=lambda x: x[0]
    )


# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for p in sorted(averaged.keys()):
    ns, accs = zip(*averaged[p])
    ax.plot(ns, accs, marker="o", label=f"p = {p}")

ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Perfect (ratio = 1)")

ax.set_xlabel("Number of Items", fontsize=12)
ax.set_ylabel("Accuracy  (quantum / classical)", fontsize=12)
ax.set_title("QAOA Accuracy vs Problem Size, iterations = 8192, 10", fontsize=14)
ax.legend(title="QAOA layers", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_ylim(0, 1.1)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("qaoa_accuracy_10_8192.png", dpi=150, transparent=True)
plt.show()
print("Saved qaoa_accuracy.png")