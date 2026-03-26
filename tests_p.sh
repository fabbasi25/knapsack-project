#!/bin/bash
#SBATCH --job-name=knapsack_p
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-99
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

INPUT_DIR="/knapsack-project/instances/easy"
OUTPUT_DIR="/knapsack-project/solutions/easy"

mkdir -p $OUTPUT_DIR
mkdir -p logs

# get list of files
FILES=($INPUT_DIR/*.txt)

INPUT_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# safety check
if [ -z "$INPUT_FILE" ]; then
    echo "No file for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

BASENAME=$(basename "$INPUT_FILE" .txt)
OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}-f.txt"

echo "Processing $INPUT_FILE"

python tests_p.py "$INPUT_FILE" "$OUTPUT_FILE"