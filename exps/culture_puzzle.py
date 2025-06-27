"""
Culture Puzzle Training with GRPO

This script trains a language model to solve culture grid transformation puzzles
using Group Relative Policy Optimization (GRPO) with the verifiers framework.

The puzzles involve learning transformations between 10x10 grids with colored cells,
similar to the interpretability-culture experiments by François Fleuret.

Usage:
# Start vLLM inference server
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model 'Qwen/Qwen2.5-3B-Instruct' --tensor-parallel-size 4

# Run training
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num-processes 4 --config-file verifiers/configs/zero3.yaml exps/culture_puzzle.py
"""

import verifiers as vf
from datasets import load_dataset
from rlvf.culture_puzzles import ascii_to_grid
import torch

size = "3B"
model_name = f"Qwen/Qwen2.5-{size}-Instruct"
run_name = f"culture-puzzle-grpo-{size}"

## Dataset and Helper Functions

parser = vf.XMLParser(["think", "answer"])

system_prompt = f"""Think step-by-step inside <think>...</think> tags, then provide your answer as a 10x10 grid inside <answer>...</answer> tags.

- Use '.' for empty cells.
- Use 'A' through 'J' for values.
- Separate each cell with a space.
- Provide exactly 10 rows of 10 cells each.
- You must include BOTH <think> and <answer> tags.

## How to solve

There is a single transformation applied to grid A to map it to f(A).

- Look for patterns in how cells change from A to f(A).
- What cells in grid B would change as a result of the transformation f?

## Example

<think>
Think step-by-step...
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
</answer>"""

def format_culture_puzzle(batch):
    def _single(example):
        return {
            "question": example["question"],
            "answer": example["grids"],
        }

    if isinstance(batch, list):
        return list(map(_single, batch))
    else:
        return _single(batch)

dataset = load_dataset("tommyp111/culture-puzzles-10k-prompt", split="forward")
dataset = dataset.map(format_culture_puzzle, batched=True, num_proc=12)
dataset.set_format("torch")


eval_size = 10
dd = dataset.train_test_split(test_size=64)
train_dataset = dd["train"]
eval_dataset = dd["test"]


def changed_cells_similarity(original, ground_truth, predicted) -> float:
    """
    Calculate similarity focusing on cells that should change vs actually changed.

    This measures how well the predicted transformation matches the ground truth
    by considering both:
    1. Cells that were supposed to change (original != ground_truth)
    2. Cells that actually changed in prediction (original != predicted)

    Args:
        original: Original grid before transformation
        ground_truth: Correct transformation result
        predicted: Predicted transformation result

    Returns:
        Similarity score based on change pattern matching (1.0 = perfect change pattern match)
        Returns 1.0 if no cells were supposed to change and none did change
    """
    if not (original.shape == ground_truth.shape == predicted.shape):
        return 0.0

    # Find cells that were supposed to change (ground truth transformation)
    should_change_mask = original != ground_truth

    # Find cells that actually changed (predicted transformation)
    did_change_mask = original != predicted

    # Union of all cells that either should change or did change
    relevant_mask = should_change_mask | did_change_mask
    relevant_count = relevant_mask.sum()

    # If no cells in either category, perfect match
    if relevant_count == 0:
        return 1.0

    # For cells in the relevant set, check if prediction matches ground truth
    matching_relevant = ((ground_truth == predicted) & relevant_mask).sum()

    return (matching_relevant / relevant_count).item()

def _parse_answer(completion):
    predicted = parser.parse_answer(completion) or ""
    if predicted is None: return None
    predicted_grid = ascii_to_grid(predicted.strip())
    return predicted_grid

def valid_grid_reward(completion, answer, **kwargs):
    predicted_grid = _parse_answer(completion)
    return 1.0 if predicted_grid is not None else 0.0


def correct_cell_reward(completion, answer, **kwargs):
    predicted_grid = _parse_answer(completion)
    if predicted_grid is None: return 0.0

    original = answer["B"]
    ground_truth = answer["f(B)"]
    return changed_cells_similarity(original, ground_truth, torch.tensor(predicted_grid))

def exact_match_reward(completion, answer, **kwargs):
    predicted_grid = _parse_answer(completion)
    if predicted_grid is None: return 0.0
    is_match = torch.all(predicted_grid == answer["f(B)"]).item()
    return 1.0 if is_match else 0.0

rubric = vf.Rubric(
    funcs=[exact_match_reward, correct_cell_reward, valid_grid_reward, parser.get_format_reward_func()],
    weights=[1, 0.6, 0.2, 0.1],  # (exact match, correct cells, valid grid, format)
)

env = vf.SingleTurnEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)
model, tokenizer = vf.get_model_and_tokenizer(model_name)

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 8
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 4
training_args.max_prompt_length = 1024
training_args.max_completion_length = 2048
training_args.max_steps = 1000
training_args.eval_steps = 200
training_args.save_steps = 200
training_args.logging_steps = 2

trainer = vf.GRPOTrainer(
    model=model, processing_class=tokenizer, env=env, args=training_args
)

trainer.train()

model.push_to_hub(f"tommyp111/{run_name}")
tokenizer.push_to_hub(f"tommyp111/{run_name}")
