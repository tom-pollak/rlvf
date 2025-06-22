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
from 

size = "1.5B"
model_name = f"Qwen/Qwen2.5-{size}-Instruct"
run_name = f"culture-puzzle-grpo-{size}"

## Dataset and Helper Functions

parser = vf.XMLParser(["think", "answer"])

system_prompt = f"""You are an expert at solving grid transformation puzzles. Given a pattern showing how one grid transforms into another, apply the same transformation to a new grid.

Think step-by-step inside <think>...</think> tags, then provide your answer as a 10x10 grid inside <answer>...</answer> tags.

- Use '.' for empty cells
- Use 'A' through 'J' for values
- Separate each cell with a space
- Provide exactly 10 rows of 10 cells each

## Example Format:

<think>
I need to analyze the transformation from A to f_A:
- Look for patterns in how cells change
- Identify the transformation
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


def changed_cells_similarity(
    original: np.ndarray, ground_truth: np.ndarray, predicted: np.ndarray
) -> float:
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
    relevant_count = np.sum(relevant_mask)

    # If no cells in either category, perfect match
    if relevant_count == 0:
        return 1.0

    # For cells in the relevant set, check if prediction matches ground truth
    matching_relevant = np.sum((ground_truth == predicted) & relevant_mask)

    return matching_relevant / relevant_count

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
