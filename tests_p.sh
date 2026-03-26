#!/bin/bash
#SBATCH --job-name=knapsack_p
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err
#SBATCH --array=0-47
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

source /home/fnabbasi/projects/def-mh541-ab/fnabbasi/qiskit_env/bin/activate 

INPUT_DIR="/home/fnabbasi/projects/def-mh541-ab/fnabbasi/knapsack-project/instances/memory_baseline"
OUTPUT_DIR="/home/fnabbasi/projects/def-mh541-ab/fnabbasi/knapsack-project/solutions/memory_baseline"

mkdir -p $OUTPUT_DIR
mkdir -p logs

# get list of files
FILES=($INPUT_DIR/*.knap)

INPUT_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# safety check
if [ -z "$INPUT_FILE" ]; then
    echo "No file for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

BASENAME=$(basename "$INPUT_FILE" .knap)
OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}-f.txt"

echo "Processing $INPUT_FILE"

python tests_p.py "$INPUT_FILE" "$OUTPUT_FILE"
