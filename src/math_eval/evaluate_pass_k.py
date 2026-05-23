import argparse
import json
import time
from pathlib import Path

import pandas as pd
from pandarallel import pandarallel

# Local imports
from parser import extract_answer, strip_string
from grader import math_equal
from python_executor import run_llm_code, SandboxExecutor
from pass_k_utils import estimate_pass_at_k

# Global execution sandbox
executor = SandboxExecutor()
def execute_and_extract_answer(response: str, data_name: str) -> str:
    """Executes LLM code if present, otherwise extracts standard text answer."""
    # Fix: Ensure response is string before "in" checks to avoid errors if float passed
    if "<llm-code>" in str(response):
        code_output = run_llm_code(response, executor)
        result = ''
        if code_output and len(code_output) > 0:
            result = code_output[-1].get('result', '')
        return result
    else:
        return extract_answer(str(response), data_name=data_name)

def load_samples(sample_dir: Path, split: str) -> pd.DataFrame:
    """Loads and concatenates all generation CSVs into a single DataFrame."""
    all_generation_csv = list(sample_dir.glob(f"{split}.generations*.csv"))
    if not all_generation_csv:
        raise FileNotFoundError(f"No generation CSVs found in {sample_dir} for split '{split}'")
    
    all_generation_df = [pd.read_csv(p) for p in all_generation_csv]
    return pd.concat(all_generation_df, ignore_index=True)

def process_df(df: pd.DataFrame, data_name: str, is_numeric = False) -> pd.DataFrame:
    """Applies extraction, length counting, and correctness grading in parallel."""
    # Fix: Ensure responses are string before split()
    df['response'] = df['response'].astype(str)
    df['response_len'] = df['response'].apply(lambda x: len(str(x).split()))
    df['pred_answer'] = df['response'].parallel_apply(lambda x: execute_and_extract_answer(x, data_name=data_name))
    df['gt_answer'] = df['gt_answer'].apply(strip_string)
    df['is_valid'] = df['pred_answer'].apply(lambda x: len(str(x)) > 0)
    # Grade answers
    df['is_correct'] = df.parallel_apply(
        lambda row: math_equal(row['pred_answer'], row['gt_answer'], timeout=False, is_numeric=is_numeric), axis=1
    )
    return df

def compute_passk(df: pd.DataFrame, cut_off_size: int = 256):
    """Computes detailed and mean pass@k metrics up to the minimum sample count."""
    grouped_df = df.groupby(['question_id', 'prompt_id'])
    
    # Aggregate correctness lists and truncate to cut_off_size
    corrects_grouped = grouped_df['is_correct'].apply(list).reset_index(name='corrects')
    corrects_grouped['corrects'] = corrects_grouped['corrects'].apply(lambda x: x[:cut_off_size])
    
    corrects_grouped['num_samples'] = corrects_grouped['corrects'].apply(len)
    corrects_grouped['num_math_equal'] = corrects_grouped['corrects'].apply(sum)
    corrects_grouped = corrects_grouped.sort_values(by=['question_id'])

    min_num_samples = corrects_grouped['num_samples'].min()
    
    # Dynamically bound k to avoid unnecessary computation
    max_k = min(2000, min_num_samples)
    
    detail_pass_at_k = {}
    for k in range(1, max_k + 1):
        detail_pass_at_k[f"pass@{k}"] = estimate_pass_at_k(
            corrects_grouped['num_samples'].values, 
            corrects_grouped['num_math_equal'].values, 
            k
        )
        
    pass_at_k = {k: detail_pass_at_k[k].mean() for k in detail_pass_at_k}
    return detail_pass_at_k, pass_at_k

def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k metrics for LLM generations.")
    
    # Add CLI arguments
    parser.add_argument(
        "--dirs", 
        nargs="+", 
        required=True, 
        help="One or more directories containing the generation CSV files."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="math", 
        help="The name of the dataset for the extraction parser (default: math)."
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="gsm8k_test", 
        help="The dataset split prefix for the files, e.g., 'gsm8k_test' (default: gsm8k_test)."
    )
    parser.add_argument(
        "--cutoff", 
        type=int, 
        default=256, 
        help="Maximum number of samples to consider per question for pass@k (default: 256)."
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=32, 
        help="Number of parallel workers for pandarallel (default: 32)."
    )

    parser.add_argument(
        "--is_numeric",
        action="store_true",
        default=False,
        help="Set this flag if responses should be interpreted as numeric only (default: False)."
    )
    
    args = parser.parse_args()

    # Initialize parallel processing based on CLI args
    pandarallel.initialize(progress_bar=True, nb_workers=args.workers)

    for dir_path_str in args.dirs:
        config_dir = Path(dir_path_str)
        print(f"Processing: {config_dir}")
        
        try:
            # 1. Load & Process Data
            generation_df = load_samples(config_dir, args.split)
            generation_df = process_df(generation_df, data_name=args.dataset, is_numeric=args.is_numeric)
            
            # 2. Save Processed Data
            processed_csv_path = config_dir / f"{args.split}.processed_generation.csv"
            generation_df.to_csv(processed_csv_path, index=False)
            
            # 3. Compute Metrics
            detail_pass_at_k, pass_at_k = compute_passk(generation_df, cut_off_size=args.cutoff)
            
            overall_results = {
                'detail_pass_at_k': {k: v.tolist() for k, v in detail_pass_at_k.items()},
                'pass_at_k': pass_at_k,
                'mean_response_len': float(generation_df['response_len'].mean()),
            }
            final_metrics = {'overall': overall_results}

            # 4. Save Metrics with sortable timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            metrics_json_path = config_dir / f'{args.split}.metrics.{timestamp}.json'
            
            with open(metrics_json_path, mode='w', encoding='utf-8') as json_file:
                json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)
                
            print(f"Metrics saved to {metrics_json_path}\n")
            
        except Exception as e:
            print(f"Error processing {config_dir}: {e}")

if __name__ == "__main__":
    main()