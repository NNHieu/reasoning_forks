import argparse
import copy
import json
import os
import random
import re
import string

import datasets
import sympy as sp

# ==========================================
# Core Problem Generation Functions
# ==========================================

def generate_base_problem(math_world, num_constants, premise_nodes, target_node, max_const_value=20):
    """Generates the underlying math rules, premises, and letter mappings without solving it."""
    nodes = list(set(re.findall(r"(<\|[a-zA-Z0-9_]+\|>)", math_world)))

    # Assign single letters to nodes
    letters = list(string.ascii_lowercase)
    if len(nodes) > len(letters):
        raise ValueError("Too many nodes for single-letter mapping. Need a broader naming scheme.")
    
    random.shuffle(letters)
    node_to_letter = {node: letters[i] for i, node in enumerate(sorted(nodes))}
    
    # Initialize constants if needed
    if num_constants > 0:
        inited_math_world = math_world.format(*[random.randint(1, max_const_value) for _ in range(num_constants)])
    else:
        inited_math_world = math_world
        
    # Replace node tokens with assigned letters
    for node in nodes:
        inited_math_world = inited_math_world.replace(node, node_to_letter[node])
    
    rules = {}
    value_dict = {}
    
    # Parse rules and known values
    for r in inited_math_world.split(","):
        lhs, rhs = r.split("=")
        if rhs.strip().isnumeric():
            value_dict[lhs] = int(rhs)
            rules[lhs] = rhs
        rules[lhs] = str(sp.sympify(rhs))

    for n in premise_nodes:
        node_letter = node_to_letter[n]
        value_dict[node_letter] = random.randint(1, max_const_value)
        rules[node_letter] = str(value_dict[node_letter])

    premises = {node_to_letter[n]: value_dict[node_to_letter[n]] for n in premise_nodes}
    target = node_to_letter[target_node]

    return {
        "rules": rules,
        "premises": premises,
        "target": target,
        "node_to_letter": node_to_letter,
        "value_dict_init": copy.deepcopy(value_dict)
    }

opening_sentences = [
    "To find the target value, we compute the following variables step by step:",
    "We compute the following variables step by step to obtain the target value.",
    "The target value is found by sequentially computing the following variables.",
]

def generate_forward_solution_trace(base_problem, solution_nodes, with_rationale=None):
    """Generates a forward reasoning trace for a given base problem."""
    rules = base_problem["rules"]
    value_dict = copy.deepcopy(base_problem["value_dict_init"])
    target = base_problem["target"]
    node_to_letter = base_problem["node_to_letter"]

    steps = []
    for step_node in solution_nodes:
        var_name = node_to_letter[step_node]
        step_exp = rules[var_name]
        var_value = int(sp.sympify(step_exp).subs(value_dict))
        value_dict[var_name] = var_value
        steps.append((var_name, step_exp, var_value))
    target_value = value_dict[target]

    lines = []
    if with_rationale == "reverse":
        reversed_vars = [s[0] for s in reversed(steps)][1:]
        reversed_vars = ", ".join(reversed_vars)
        lines.append(f"<rationale>Since {target} is depended on {reversed_vars}, I choose branch {reversed_vars[-1]}.</rationale>")
    elif with_rationale is not None:
        raise ValueError(f"Unknown with_rationale={with_rationale!r}")

    lines.append(random.choice(opening_sentences))
    for i, (var, expr, val) in enumerate(steps, 1):
        lines.append(f"{i}. {var} = {expr} = {val}.")
    lines.append(f"\nThus, {steps[-1][0]} = \\boxed{{{steps[-1][2]}}}.")

    return {
        "problem": {
            "rules": rules,
            "premises": base_problem["premises"],
            "target": target
        },
        "answer": {
            "steps": steps,
            "target_var_name": target,
            "target_value": target_value,
            "value_dict": value_dict,
            "is_reverse": False,
        },
        "answer_description": "\n".join(lines),
    }

def generate_reverse_solution_trace(base_problem, solution_nodes):
    """Generates a reverse reasoning trace for a given base problem."""
    rules = base_problem["rules"]
    value_dict = copy.deepcopy(base_problem["value_dict_init"])
    target = base_problem["target"]
    node_to_letter = base_problem["node_to_letter"]

    steps = []
    expr_dict = {}
    target_expression = sp.sympify(rules[target])
    for step_node in solution_nodes:
        var_name = node_to_letter[step_node]
        step_exp = rules[var_name]
        expr_dict[var_name] = sp.sympify(step_exp)
        target_expression = target_expression.subs(expr_dict)
        steps.append((var_name, step_exp, str(target_expression)))
    target_value = int(target_expression)

    lines = [random.choice(opening_sentences)]
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. Substitute {step[0]} = {step[1]} into the target expression, yielding {target} = {step[2]}.")
    lines.append(f"\nThus, {target} = \\boxed{{{steps[-1][2]}}}.")

    return {
        "problem": {
            "rules": rules,
            "premises": base_problem["premises"],
            "target": target
        },
        "answer": {
            "steps": steps,
            "target_var_name": target,
            "target_value": target_value,
            "value_dict": value_dict,
            "is_reverse": True,
        },
        "answer_description": "\n".join(lines),
    }

# ==========================================
# Text Formatting Functions
# ==========================================

RELATION_TEMPLATES = ["{} = {}"]
GIVEN_TEMPLATES = ["{} = {}"]
PROBLEM_TEMPLATES = [
    lambda rels, givens, query: (
        f"Let each letter represent a numerical variable. These variables are defined as follows: {rels}. "
        f"{f'If {givens}, what is the resulting value of {query}?' if givens else f'What is the resulting value of {query}?'}"
    ),
    lambda rels, givens, query: (
        f"Consider a system of variables where each variable is defined as follows: {rels}. "
        f"{f'If {givens}, determine the value of {query}.' if givens else f'Determine the value of {query}.'}"
    ),
]

def generate_problem_description(problem: dict, shuffle=True):
    relation_descriptions = [RELATION_TEMPLATES[0].format(k, expr) for k, expr in problem['rules'].items()]
    relation_descriptions = '; '.join(relation_descriptions)
    
    given_str = None
    if problem['premises']:
        given_strs = [GIVEN_TEMPLATES[0].format(k, v) for k, v in problem['premises'].items()]
        if shuffle: 
            random.shuffle(given_strs)
        given_str = ", ".join(given_strs)

    template = PROBLEM_TEMPLATES[1]
    return template(relation_descriptions, given_str, problem['target'])

def convert_sample_to_prompt(question_prompt, sample, world_id, direction):
    return {
        "question": question_prompt,
        "answer": sample['answer_description'],
        "meta": json.dumps(sample),
        "world_id": world_id,
        "direction": direction
    }

# ==========================================
# Dataset Pipeline
# ==========================================

def generate_unique_problems(size, world):
    """Generates a pool of unique base problems (without attaching solutions yet)."""
    problems = []
    seen_rules = set() 
    
    while len(problems) < size:
        base_problem = generate_base_problem(
            world['math_world'], 
            world['num_constants'], 
            world['premise_nodes'], 
            world['target_node']
        )
        
        # O(1) uniqueness check using frozenset
        rules_frozenset = frozenset(base_problem['rules'].values())
        if rules_frozenset in seen_rules:
            continue
            
        seen_rules.add(rules_frozenset)
        problems.append(base_problem)
        
    return problems

def default_base_problems_path(data_dir, world_id):
    return os.path.join(data_dir, world_id, "base_problems.json")

def save_base_problems(path, base_problems, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"metadata": metadata, "base_problems": base_problems}, f, indent=2)

def load_base_problems(path):
    with open(path) as f:
        payload = json.load(f)
    return payload["base_problems"], payload["metadata"]

def split_base_problems(base_problems, sft_train_size, rlvr_train_size):
    return (
        base_problems[:sft_train_size],
        base_problems[sft_train_size:sft_train_size + rlvr_train_size],
        base_problems[sft_train_size + rlvr_train_size:],
    )

def build_prompt_samples(sft_train_base, rlvr_train_base, test_base, world_config):
    """Attach solution traces and natural-language prompts to base problems."""
    sft_train_forward_samples = []
    sft_train_reverse_samples = []
    sft_train_forward_withrationale_samples = []
    rlvr_train_samples = []

    for p in sft_train_base:
        question_prompt = generate_problem_description(p, shuffle=True)
        fwd_trace = generate_forward_solution_trace(p, world_config['forward_solution'])
        sft_train_forward_samples.append(
            convert_sample_to_prompt(question_prompt, fwd_trace, world_config['id'], 'forward')
        )

        fwd_withrationale_trace = generate_forward_solution_trace(p, world_config['forward_solution'], with_rationale="reverse")
        sft_train_forward_withrationale_samples.append(
            convert_sample_to_prompt(question_prompt, fwd_withrationale_trace, world_config['id'], 'forward_with_reverse_rationale')
        )

        rev_trace = generate_reverse_solution_trace(p, world_config['reverse_solution'])
        sft_train_reverse_samples.append(
            convert_sample_to_prompt(question_prompt, rev_trace, world_config['id'], 'reverse')
        )

    for p in rlvr_train_base:
        question_prompt = generate_problem_description(p, shuffle=True)
        rlvr_trace = generate_forward_solution_trace(p, world_config['forward_solution'])
        rlvr_train_samples.append(
            convert_sample_to_prompt(question_prompt, rlvr_trace, world_config['id'], 'forward')
        )

    random.shuffle(sft_train_forward_samples)
    random.shuffle(sft_train_forward_withrationale_samples)
    random.shuffle(sft_train_reverse_samples)
    random.shuffle(rlvr_train_samples)

    test_samples = []
    for p in test_base:
        question_prompt = generate_problem_description(p, shuffle=True)
        fwd_trace = generate_forward_solution_trace(p, world_config['forward_solution'])
        test_samples.append(
            convert_sample_to_prompt(question_prompt, fwd_trace, world_config['id'], 'forward')
        )

    return {
        "sft_train_forward": sft_train_forward_samples,
        "sft_train_reverse": sft_train_reverse_samples,
        "sft_train_forward_with_reverse_rationale": sft_train_forward_withrationale_samples,
        "rlvr_train": rlvr_train_samples,
        "test": test_samples,
    }

def save_prompt_datasets(samples: dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    processed = {}
    for name, raw_samples in samples.items():
        raw_ds = datasets.Dataset.from_list(raw_samples)
        processed[name] = raw_ds.map(function=make_map_fn(name), with_indices=True)
        processed[name].to_parquet(os.path.join(output_dir, f"{name}.parquet"))

    with open(os.path.join(output_dir, "train_example.json"), "w") as f:
        json.dump({
            "forward_example": processed["sft_train_forward"][0],
            "reverse_example": processed["sft_train_reverse"][0],
            "forward_with_reverse_rationale_example": processed["sft_train_forward_with_reverse_rationale"][0],
        }, f, indent=4)

    processed_prompt_data = []
    for idx, e in enumerate(processed["test"]):
        processed_prompt_data.append({
            "id": idx,
            "Question": e["question"],
            "answer": e['answer'],
            "subset": e['extra_info']['data_source'],
        })
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(processed_prompt_data, f, indent=4)

    return processed

def make_map_fn(split):
    def process_fn(example, idx):
        meta = json.loads(example['meta'])
        question = example["question"]
        return {
            "question": question,
            "solution": example["answer"],
            "answer": meta["answer"]["target_value"],
            "extra_info": {
                "data_source": example['world_id'],
                "direction": example["direction"],
                "split": split,
                "index": idx,
            },
        }
    return process_fn

# ==========================================
# Main Execution
# ==========================================

WORLDS = [
    {
        "id": "arithchain_2_10",
        "math_world": "<|a1|>=<|p0|>+{},<|a2|>=<|a1|>+{},<|a3|>=<|a2|>+{},<|a4|>=<|a3|>+{},<|a5|>=<|a4|>+{},<|a6|>=<|a5|>+{},<|a7|>=<|a6|>+{},<|a8|>=<|a7|>+{},<|a9|>=<|a8|>+{},<|a10|>=<|a9|>+{},<|b1|>=<|p0|>+{},<|b2|>=<|b1|>+{},<|b3|>=<|b2|>+{},<|b4|>=<|b3|>+{},<|b5|>=<|b4|>+{},<|b6|>=<|b5|>+{},<|b7|>=<|b6|>+{},<|b8|>=<|b7|>+{},<|b9|>=<|b8|>+{},<|b10|>=<|b9|>+{}",
        "num_constants": 20,
        "premise_nodes": ("<|p0|>", ),
        "target_node": "<|a10|>",
        "forward_solution": ("<|a1|>", "<|a2|>", "<|a3|>", "<|a4|>", "<|a5|>", "<|a6|>", "<|a7|>", "<|a8|>", "<|a9|>", "<|a10|>"),
        "reverse_solution": ('<|a9|>', '<|a8|>', '<|a7|>', '<|a6|>', '<|a5|>', '<|a4|>', '<|a3|>', '<|a2|>', '<|a1|>', '<|p0|>')
    }
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic math world datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two-stage workflow:
  # Stage 1: generate and persist unique base problems
  python src/data_generation/gen_arithchain.py --stage base

  # Stage 2: load base problems and write prompt/answer parquets
  python src/data_generation/gen_arithchain.py --stage prompts --seed 42

  # Or run both stages in one command (default)
  python src/data_generation/gen_arithchain.py --stage all
""",
    )
    parser.add_argument(
        "--stage",
        choices=["base", "prompts", "all"],
        default="all",
        help="Pipeline stage: base (unique problems only), prompts (from saved base), or all",
    )
    parser.add_argument("--world_index", type=int, default=0, help="Index of the world config to use from WORLDS")
    parser.add_argument("--sft_train_size", type=int, default=6400, help="Number of unique base SFT training samples")
    parser.add_argument("--rlvr_train_size", type=int, default=1600, help="Number of unique base RLVR training samples")
    parser.add_argument("--test_size", type=int, default=1000, help="Number of unique base test samples")
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Output directory for parquets and base problems")
    parser.add_argument(
        "--base_problems_path",
        type=str,
        default=None,
        help="Path to base_problems.json (default: <data_dir>/<world_id>/base_problems.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    world_config = WORLDS[args.world_index]
    base_problems_path = args.base_problems_path or default_base_problems_path(args.data_dir, world_config["id"])
    output_dir = os.path.join(args.data_dir, world_config["id"])
    total_size = args.sft_train_size + args.rlvr_train_size + args.test_size
    metadata = {
        "world_index": args.world_index,
        "world_id": world_config["id"],
        "sft_train_size": args.sft_train_size,
        "rlvr_train_size": args.rlvr_train_size,
        "test_size": args.test_size,
        "total_size": total_size,
        "seed": args.seed,
    }

    print(f"Processing world: {world_config['id']} (stage={args.stage})...")

    if args.stage in ("base", "all"):
        random.seed(args.seed)
        print(f"Generating {total_size} unique base problems...")
        base_problems = generate_unique_problems(total_size, world_config)
        save_base_problems(base_problems_path, base_problems, metadata)
        print(f"Saved {len(base_problems)} base problems to {base_problems_path}")

    if args.stage in ("prompts", "all"):
        if args.stage == "prompts":
            if not os.path.isfile(base_problems_path):
                raise FileNotFoundError(
                    f"Base problems file not found: {base_problems_path}. "
                    "Run with --stage base first."
                )
            base_problems, saved_metadata = load_base_problems(base_problems_path)
            if saved_metadata["world_index"] != args.world_index:
                raise ValueError(
                    f"world_index {args.world_index} does not match saved "
                    f"world_index {saved_metadata['world_index']} in {base_problems_path}"
                )
            expected_total = (
                saved_metadata["sft_train_size"]
                + saved_metadata["rlvr_train_size"]
                + saved_metadata["test_size"]
            )
            if len(base_problems) != expected_total:
                raise ValueError(
                    f"Expected {expected_total} base problems, found {len(base_problems)} "
                    f"in {base_problems_path}"
                )
            sft_train_size = saved_metadata["sft_train_size"]
            rlvr_train_size = saved_metadata["rlvr_train_size"]
            print(f"Loaded {len(base_problems)} base problems from {base_problems_path}")
        else:
            sft_train_size = args.sft_train_size
            rlvr_train_size = args.rlvr_train_size
            base_problems, _ = load_base_problems(base_problems_path)

        random.seed(args.seed)
        sft_train_base, rlvr_train_base, test_base = split_base_problems(
            base_problems, sft_train_size, rlvr_train_size
        )
        samples = build_prompt_samples(sft_train_base, rlvr_train_base, test_base, world_config)
        save_prompt_datasets(
            samples,
            output_dir,
        )
        # Print sample counts for all splits
        for split_name, split_samples in samples.items():
            print(f"{split_name}: {len(split_samples)} samples")
        print(
            f"Pipeline complete. All splits written to {output_dir}"
        )
 
   