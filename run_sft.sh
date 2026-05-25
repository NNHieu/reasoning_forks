#!/usr/bin/env bash

DATASET_NAME=$1
model_short_name=$2
NUM_TRAIN_EPOCHS=$3
BATCH_SIZE_ARG=$4
shift 4

DATA_ROOT="/scratch1/hnn5071/workspace/rm-limeval/datasets"

########################
# MODEL CONFIG
########################
declare -A MODEL_CONFIGS

MODEL_CONFIGS["qwen2.5_0.5b"]="unsloth/Qwen2.5-0.5B 32 1 1e-5"
MODEL_CONFIGS["evolm-1b"]="zhenting/evolm-1B-160BT-cpt-MixedFW8FM42 32 1 1e-5"
MODEL_CONFIGS["evolm-4b"]="zhenting/evolm-4B-160BT-cpt-MixedFW8FM42 32 1 1e-5"

# Validate model
if [ -z "${model_short_name}" ]; then
    echo "Error: model_short_name is not set"
    exit 1
fi

CONFIG="${MODEL_CONFIGS[$model_short_name]}"

if [ -z "$CONFIG" ]; then
    echo "Invalid model name. Options are:"
    printf "  - %s\n" "${!MODEL_CONFIGS[@]}"
    exit 1
fi

# Parse model config
read -r MODEL_NAME DEFAULT_BATCH_SIZE GRAD_ACCUM LR <<< "$CONFIG"

# Use batch size from argument if provided, else default
if [ -z "$BATCH_SIZE_ARG" ]; then
    BATCH_SIZE="$DEFAULT_BATCH_SIZE"
else
    BATCH_SIZE="$BATCH_SIZE_ARG"
fi

TOTAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

########################
# DATASET CONFIG
########################
declare -A DATA_PATHS
declare -A DATA_SIZES   # for dynamic SAVE_STEPS

DATA_PATHS["arithchain_2_10_forward"]="datasets/arithchain_2_10/sft_train_forward.parquet"
DATA_PATHS["arithchain_2_10_reverse"]="datasets/arithchain_2_10/sft_train_reverse.parquet"
DATA_PATHS["arithchain_2_10_forward_with_reverse_rationale"]="datasets/arithchain_2_10/sft_train_forward_with_reverse_rationale.parquet"

DATA_PATHS["gsm8k_datasetlevel"]="nnheui/reasoning_modes|gsm8k_train_double_datasetlevel"
DATA_PATHS["gsm8k_problemlevel"]="nnheui/reasoning_modes|gsm8k_train_double_problemlevel"

DATA_SIZES["arithchain_2_10_forward"]=6400
DATA_SIZES["arithchain_2_10_reverse"]=6400
DATA_SIZES["gsm8k_datasetlevel"]=12800
DATA_SIZES["gsm8k_problemlevel"]=12800

# Validate dataset
if [ -z "${DATASET_NAME}" ]; then
    echo "Error: DATASET_NAME is not set"
    exit 1
fi

DATA_PATH="${DATA_PATHS[$DATASET_NAME]}"
DATA_SIZE="${DATA_SIZES[$DATASET_NAME]}"

if [ -z "$DATA_PATH" ]; then
    echo "Invalid dataset name. Options are:"
    printf "  - %s\n" "${!DATA_PATHS[@]}"
    exit 1
fi

########################
# SAVE STEPS LOGIC
########################

# Default: compute from dataset size
SAVE_STEPS=$((DATA_SIZE / TOTAL_BATCH_SIZE))

# Override for fixed cases (to match your original logic exactly)
case "${DATASET_NAME}" in
  "mathgsm8k_nlreasoning" | "mathgsm8k_code")
    SAVE_STEPS=186
    ;;
  "gsm8k_nlreasoning" | "gsm8k_code" | "gsm8k_datasetlevelcombined")
    SAVE_STEPS=101
    ;;
  "gsm8k_single_mix")
    # already computed dynamically (6400 / TOTAL_BATCH_SIZE)
    ;;
esac

########################
# DEBUG PRINT
########################
echo "MODEL_NAME: $MODEL_NAME"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "LR: $LR"
echo "TOTAL_BATCH_SIZE: $TOTAL_BATCH_SIZE"

echo "DATASET_NAME: $DATASET_NAME"
echo "DATA_PATH: $DATA_PATH"
echo "DATA_SIZE: $DATA_SIZE"
echo "SAVE_STEPS: $SAVE_STEPS"

WANDB_PROJECT="reasoning_forks_sft"
RUN_NAME="${model_short_name}_sft_${DATASET_NAME}_lr${LR}_bs${BATCH_SIZE}_ga${GRAD_ACCUM}"

OUTPUT_DIR="runs/${WANDB_PROJECT}/${RUN_NAME}"

# --------------------------------------------------
# Environment
# --------------------------------------------------

export TOKENIZERS_PARALLELISM=false

mkdir -p ${OUTPUT_DIR}

LOG_FILE="${OUTPUT_DIR}/train.log"

echo "Starting training..."
echo "Run name: ${RUN_NAME}"
echo "Logs: ${LOG_FILE}"

# --------------------------------------------------
# Launch
# --------------------------------------------------

python src/training/sft.py \
  --model_name ${MODEL_NAME} \
  --data_path ${DATA_PATH} \
  --prompt_key "question" \
  --response_key "solution" \
  --chat_template_path src/alpaca_template.jira \
  --batch_size ${BATCH_SIZE} \
  --grad_accum ${GRAD_ACCUM} \
  --warmup_ratio 0.1 \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --learning_rate ${LR} \
  --save_steps ${SAVE_STEPS} \
  --output_dir ${OUTPUT_DIR} \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run_name ${RUN_NAME} $@ \
  2>&1 | tee ${LOG_FILE}

echo "Training finished."