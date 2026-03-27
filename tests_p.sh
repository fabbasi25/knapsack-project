#!/bin/bash
#SBATCH --job-name=knapsack_p
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err
#SBATCH --array=0-17
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

source /home/fnabbasi/projects/def-mh541-ab/fnabbasi/qiskit_env/bin/activate 

OUTPUT_DIR="/home/fnabbasi/projects/def-mh541-ab/fnabbasi/knapsack-project/solutions/small_5"

mkdir -p $OUTPUT_DIR

n=$((SLURM_ARRAY_TASK_ID + 1))
OUTPUT_FILE="$OUTPUT_DIR/${n}-f.txt"

echo "Running task $SLURM_ARRAY_TASK_ID (n=$n)"

python tests_p.py "$OUTPUT_FILE" "$n"