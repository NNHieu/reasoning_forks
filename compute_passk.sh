#!/bin/bash

JOB_FILE="job_list.txt"
WORKERS=16

# Default: do not set --is_numeric
IS_NUMERIC=""

# If the script is called with "--is_numeric", set the variable.
for arg in "$@"
do
    if [ "$arg" = "--is_numeric" ]; then
        IS_NUMERIC="--is_numeric"
        break
    fi
done

declare -A GROUPED_DIRS

# -----------------------------
# Step 1: Group dirs by dataset split
# key = JOB_DIR|SPLIT
# value = list of directories
# -----------------------------
while IFS="|" read -r JOB_DIR RUN_NAME SPLIT NUM_SAMPLES; do
    KEY="${JOB_DIR}|${SPLIT}"
    DIR="${JOB_DIR}/${RUN_NAME}"

    GROUPED_DIRS["$KEY"]+="$DIR "
done < "$JOB_FILE"

# -----------------------------
# Step 2: Run evaluation per group
# -----------------------------
for KEY in "${!GROUPED_DIRS[@]}"; do
    IFS="|" read -r JOB_DIR SPLIT <<< "$KEY"

    DIRS=${GROUPED_DIRS[$KEY]}

    echo "========================================"
    echo "Running pass@k for split: $SPLIT"
    echo "Dirs: $DIRS"
    echo "is_numeric: $IS_NUMERIC"
    echo "========================================"

    python src/math_eval/evaluate_pass_k.py \
        --dirs $DIRS \
        --dataset math \
        --split "${SPLIT}" \
        --workers $WORKERS \
        $IS_NUMERIC
done