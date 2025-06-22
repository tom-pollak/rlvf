"""
Culture Puzzle Training with GRPO

This script trains a language model to solve culture grid transformation puzzles
using Group Relative Policy Optimization (GRPO) with the verifiers framework.

The puzzles involve learning transformations between 10x10 grids with colored cells,
similar to the interpretability-culture experiments by François Fleuret.

Usage:
# Start vLLM inference server
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model 'Qwen/Qwen2.5-1.5B-Instruct' --tensor-parallel-size 2

# Run training
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file verifiers/configs/zero3.yaml exps/culture_puzzle.py
"""

import verifiers as vf
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Tuple, Optional

import verifiers as vf

size = "1.5B"
model_name = f"Qwen/Qwen2.5-{size}-Instruct"
run_name = f"culture-puzzle-grpo-{size}"

## Dataset and Helper Functions


def decode_puzzle_grids(input_ids: List[int]) -> Dict[int, np.ndarray]:
    """
    Decode the 405-integer format into 4 grids.

    Returns:
        Dict mapping marker positions to 10x10 numpy arrays
    """
    input_ids = np.array(input_ids)

    # Skip the conditioning token
    tokens = input_ids[1:]

    # Find marker positions
    markers = {}
    for marker_val in [11, 12, 13, 14]:
        positions = np.where(tokens == marker_val)[0]
        if len(positions) > 0:
            markers[marker_val] = positions[0]

    grids = {}
    for marker_val, pos in markers.items():
        # Extract 100 integers after the marker
        start_idx = pos + 1
        end_idx = start_idx + 100

        if end_idx <= len(tokens):
            grid_data = tokens[start_idx:end_idx]
            grid = grid_data.reshape(10, 10)
            grids[marker_val] = grid

    return grids


def grid_to_ascii(grid: np.ndarray, color_map: Optional[Dict[int, str]] = None) -> str:
    """
    Convert a 10x10 integer grid to ASCII representation.
    """
    if color_map is None:
        # Default mapping: 0='.', 1-10='A'-'J'
        color_map = {0: "."}
        for i in range(1, 11):
            color_map[i] = chr(ord("A") + i - 1)

    lines = []
    for row in grid:
        line = " ".join(color_map.get(val, "?") for val in row)
        lines.append(line)

    return "\n".join(lines)


def ascii_to_grid(
    ascii_str: str, color_map: Optional[Dict[str, int]] = None
) -> Optional[np.ndarray]:
    """
    Convert ASCII grid representation back to 10x10 integer array.
    """
    if color_map is None:
        # Default reverse mapping
        color_map = {".": 0}
        for i in range(1, 11):
            color_map[chr(ord("A") + i - 1)] = i

    try:
        lines = ascii_str.strip().split("\n")
        if len(lines) != 10:
            return None

        grid = np.zeros((10, 10), dtype=int)
        for i, line in enumerate(lines):
            tokens = line.split()
            if len(tokens) != 10:
                return None
            for j, token in enumerate(tokens):
                if token in color_map:
                    grid[i, j] = color_map[token]
                else:
                    return None

        return grid
    except Exception:
        return None


def create_llm_prompt(input_ids: List[int]) -> Tuple[str, str]:
    """
    Create a prompt suitable for LLM training.

    Returns:
        Tuple of (question, answer)
    """
    grids = decode_puzzle_grids(input_ids)

    if len(grids) < 4:
        raise ValueError(f"Expected 4 grids, got {len(grids)}")

    markers = sorted(grids.keys())

    # Extract grids in order
    A = grids[markers[0]]
    f_A = grids[markers[1]]
    B = grids[markers[2]]
    f_B = grids[markers[3]]

    question = f"""You are solving a grid transformation puzzle. Study the pattern A → f_A, then apply the same transformation to grid B.

Grid A:
{grid_to_ascii(A)}

Grid f_A (transformation of A):
{grid_to_ascii(f_A)}

Grid B:
{grid_to_ascii(B)}

What is Grid f_B (apply the same transformation to B)?"""

    answer = grid_to_ascii(f_B)

    return question, answer


def format_culture_puzzle(batch):
    """Format culture puzzle for training"""
    formatted_examples = []

    # Assume it's a batch from dataset.map()
    examples = []
    for i in range(len(batch["input_ids"])):
        examples.append({"input_ids": batch["input_ids"][i]})
    batch = examples

    for example in batch:
        question, answer = create_llm_prompt(example["input_ids"])
        formatted_examples.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    return formatted_examples


def format_culture_puzzle_batch(batch):
    """Format culture puzzle batch for training"""
    formatted_examples = {"question": [], "answer": []}

    for input_ids in batch["input_ids"]:
        try:
            question, answer = create_llm_prompt(input_ids)
            formatted_examples["question"].append(question)
            formatted_examples["answer"].append(answer)
        except Exception as e:
            # Skip malformed examples by not appending them
            print(f"Skipping malformed example: {e}")
            continue

    return formatted_examples


def load_and_process_dataset():
    """Load and process the culture puzzles dataset"""
    print("Loading culture puzzles dataset...")
    dataset = load_dataset("tommyp111/culture-puzzles-1M", split="train")

    # Take a subset for faster experimentation (remove this line for full training)
    dataset = dataset.select(range(50000))  # Use 50k samples for testing

    print(f"Processing {len(dataset)} culture puzzles...")

    # Use batched .map() to process the dataset efficiently
    processed_dataset = dataset.map(
        format_culture_puzzle_batch,
        batched=True,
        batch_size=1000,
        num_proc=12,
        desc="Processing culture puzzles",
        remove_columns=dataset.column_names,  # Remove original columns
    )

    dd = processed_dataset.train_test_split(test_size=0.1)

    return dd


def culture_grid_reward(completion, answer, **kwargs):
    """
    Reward function that gives 1.0 for exact grid match, 0.0 otherwise.
    """
    predicted_text = parser.parse_answer(completion) or ""
    predicted_text = predicted_text.strip()
    answer = answer.strip()

    # Try to parse predicted grid
    predicted_grid = ascii_to_grid(predicted_text)
    target_grid = ascii_to_grid(answer)

    if predicted_grid is None or target_grid is None:
        return 0.0

    # Exact match
    if np.array_equal(predicted_grid, target_grid):
        return 1.0

    # Partial credit for close matches (optional)
    # Could add similarity metrics here

    return 0.0


## Train!

# Load and process the dataset
dd = load_and_process_dataset()

print("\n=== Sample Training Example ===")
sample = dd["train"][0]
print("Question:")
print(sample["question"])
print("\nAnswer:")
print(sample["answer"])

parser = vf.XMLParser(["think", "answer"])

system_prompt = f"""You are an expert at solving grid transformation puzzles. Given a pattern showing how one grid transforms into another, apply the same transformation to a new grid.

Think step-by-step inside <think>...</think> tags, then provide your answer as a 10x10 grid inside <answer>...</answer> tags.

## Example Format:

<think>
I need to analyze the transformation from A to f_A:
- Look for patterns in how cells change
- Identify the type of transformation (translation, growth, fill, etc.)
- Apply the same pattern to grid B
</think>
<answer>
. . . A A A . . . .
. . . A B A . . . .
. . . A A A . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
</answer>

{parser.get_format_str()}"""


rubric = vf.Rubric(
    funcs=[culture_grid_reward, parser.get_format_reward_func()],
    weights=[1.0, 0.1],  # (correct grid, format)
)

env = vf.SingleTurnEnv(
    dataset=dd["train"],
    eval_dataset=dd["test"],
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)
model, tokenizer = vf.get_model_and_tokenizer(model_name)

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 4
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 8
training_args.max_prompt_length = 1024
training_args.max_completion_length = 2048
training_args.max_steps = 2000
training_args.eval_steps = 200
training_args.save_steps = 400
training_args.logging_steps = 10

trainer = vf.GRPOTrainer(
    model=model, processing_class=tokenizer, env=env, args=training_args
)

trainer.train()

model.push_to_hub(f"tommyp111/{run_name}")
tokenizer.push_to_hub(f"tommyp111/{run_name}")
