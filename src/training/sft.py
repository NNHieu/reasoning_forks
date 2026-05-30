import os
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import (
    standardize_sharegpt,
    train_on_responses_only,
)
import torch
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from unsloth.models.loader_utils import prepare_device_map
from unsloth import is_bfloat16_supported

# -------------------------------------------------
# Argument Parser
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script")

    # Model
    parser.add_argument("--model_name", type=str, default="allenai/Olmo-3-1025-7B")
    # parser.add_argument("--tokenizer_name", type=str, default="allenai/Olmo-3-7B-Instruct-SFT")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--conversation_key", type=str, default=None)
    parser.add_argument("--prompt_key", type=str, default="question")
    parser.add_argument("--response_key", type=str, default="answer")
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default=None,
        help="Path to chat template file to apply. E.g., 'templates/sharegpt_chat_template.json', etc."
    )

    # Instruction
    parser.add_argument("--train_on_prompt", action="store_true")
    parser.add_argument("--instruction_part", type=str, default="<|im_start|>user\n")
    parser.add_argument("--response_part", type=str, default="<|im_start|>assistant\n")

    # Training
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    # parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    # parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Checkpointing
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # WandB
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sft-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Hardware
    # parser.add_argument("--cuda_devices", type=str, default="0,1")

    return parser.parse_args()


# -------------------------------------------------
# Load Model & Tokenizer
# -------------------------------------------------

def load_model_and_tokenizer(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    device_map, distributed = prepare_device_map()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        full_finetuning=True,
        dtype=torch.bfloat16,
        device_map=device_map,
        use_gradient_checkpointing="unsloth",
        # float32_mixed_precision = True,
    )

    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    return model, tokenizer, distributed


# -------------------------------------------------
# Dataset Preparation
# -------------------------------------------------
def load_and_prepare_dataset(tokenizer, args):
    # Check for special HuggingFace loading format: "repo,split"
    if "," in args.data_path:
        repo, split = args.data_path.split(",", 1)
        dataset = load_dataset(repo, split=split)
        print(f"Loaded HuggingFace dataset: repo={repo}, split={split}")
        # Use as-is; HF datasets returns DatasetDict if has multiple splits, else Dataset
        if isinstance(dataset, dict) and split in dataset:
            dataset = dataset[split]
    else:
        dataset = load_dataset("parquet", data_files=args.data_path)["train"]
        print("Loaded parquet dataset from:", args.data_path)
    print("dataset size: ", len(dataset))
    print(dataset[0])

    dataset = dataset.map(
        lambda e: {
            "messages": [
                {"role": "user", "content": e[args.prompt_key]},
                {"role": "assistant", "content": " " + e[args.response_key]},
            ]
        }
    )
    dataset = standardize_sharegpt(dataset)

    # Determine chat_template: Use file if path provided, else use tokenizer default
    chat_template = None
    if getattr(args, "chat_template_path", None) is not None:
        with open(args.chat_template_path, "r", encoding="utf-8") as f:
            chat_template = f.read()
    else:
        # Use the default template embedded in the tokenizer if available
        chat_template = getattr(tokenizer, "chat_template", None)

    def format_batch(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template,
            )
            for convo in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(format_batch, batched=True)

    return dataset


# -------------------------------------------------
# Training
# -------------------------------------------------

def train(args):
    # args.use_wandb = args.use_wandb and torch.distributed.get_rank() == 0
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    model, tokenizer, distributed = load_model_and_tokenizer(args)
    dataset = load_and_prepare_dataset(tokenizer, args)
    print("len processed dataset: ", len(dataset))
    print(dataset[0])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length, # Explicitly set this here
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            # warmup_steps=args.warmup_steps,
            # max_steps=args.max_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_torch_fused",
            weight_decay=0.0,
            # lr_scheduler_type="linear",
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=args.output_dir,
            ddp_find_unused_parameters=True if distributed else None,
            # Logging
            report_to="wandb" if args.use_wandb else "none",

            # Checkpointing
            save_strategy="steps",
            save_steps=args.save_steps,
            # save_total_limit=args.save_total_limit,

            dataset_num_proc=2,
            packing=False,      # Set to True only if your sequences are very short
        ),
    )

    if not args.train_on_prompt:
        trainer = train_on_responses_only(
            trainer,    
            instruction_part = "### Instruction:\n",
            response_part = "### Response:\n",
        )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained_merged(args.output_dir + "/finetune_16bit", tokenizer, save_method = "merged_16bit",)
    
    if args.use_wandb:
        wandb.finish()


# -------------------------------------------------
# Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)