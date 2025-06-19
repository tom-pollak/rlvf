"""
Chess Puzzle Training with GRPO

This script trains a language model to solve chess puzzles from the Lichess dataset
using Group Relative Policy Optimization (GRPO) with the verifiers framework.

Usage:
# Start vLLM inference server
CUDA_VISIBLE_DEVICES=0 vf-vllm --model 'Qwen/Qwen2.5-1.5B-Instruct'

# Run training
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file verifiers/configs/zero3.yaml chess_puzzle_experiment.py
"""

import verifiers as vf
from datasets import load_dataset

# Model configuration
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Load Lichess chess puzzles dataset
print("Loading chess puzzles dataset...")
dataset = load_dataset("Lichess/chess-puzzles", split="train")


# Format dataset for training
def format_chess_puzzle(example):
    """Format chess puzzle for training"""
    return {
        "question": f"FEN: {example['FEN']}\nSolve this chess puzzle.",
        "answer": example["Moves"],
    }


dataset = dataset.map(format_chess_puzzle)

# Split dataset for training and evaluation
eval_size = min(1000, len(dataset) // 10)  # 10% or 1000 examples for eval
eval_dataset = dataset.select(range(eval_size))
train_dataset = dataset.select(range(eval_size, len(dataset)))

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# Create parser for structured output
parser = vf.XMLParser(["think", "answer"])

# System prompt for chess puzzle solving
system_prompt = f"""Solve the chess puzzle by finding the best possible next move.

Think step-by-step inside <think>...</think> tags, then give your answer as a single move in UCI notation inside <answer>...</answer> tags.

## Example:

<think>
In here, you should think step-by-step about the puzzle. Analyze:
- Key pieces and their positions
- Tactical patterns (pins, forks, skewers, discoveries)
- Forcing moves (checks, captures, threats)
- The puzzle's objective (mate, win material, etc.)
</think>
<answer>
b6c5
</answer>

{parser.get_format_str()}"""


# Reward function for exact move sequence matching
def chess_moves_reward(completion, answer, **kwargs):
    """
    Reward function that gives 1.0 for exact match, 0.0 otherwise.
    Could be enhanced with partial credit for correct first moves.
    """
    predicted = parser.parse_answer(completion) or ""
    predicted = predicted.strip()
    answer = answer.strip()

    # Exact match
    if predicted == answer:
        return 1.0

    # Optional: partial credit for correct first move
    if predicted and answer:
        pred_moves = predicted.split()
        answer_moves = answer.split()
        if len(pred_moves) > 0 and len(answer_moves) > 0:
            if pred_moves[0] == answer_moves[0]:
                return 0.3  # Partial credit for correct first move

    return 0.0


# Create rubric with move accuracy and format rewards
rubric = vf.Rubric(
    funcs=[chess_moves_reward, parser.get_format_reward_func()],
    weights=[1.0, 0.2],  # Prioritize correct moves over format
)

# Create training environment
print("Creating chess puzzle environment...")
env = vf.SingleTurnEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)

# Load model and tokenizer
print(f"Loading model: {model_name}")
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Configure training arguments
training_args = vf.grpo_defaults(run_name="chess-puzzle-grpo")

# Chess-specific training configuration
training_args.max_steps = 1000
training_args.per_device_train_batch_size = 4
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 2
training_args.learning_rate = 5e-6
training_args.max_prompt_length = 1024
training_args.max_completion_length = 512
training_args.eval_steps = 100
training_args.save_steps = 200
training_args.logging_steps = 10

# Create trainer
print("Creating GRPO trainer...")
trainer = vf.GRPOTrainer(
    model=model, processing_class=tokenizer, env=env, args=training_args
)

if __name__ == "__main__":
    print("Starting chess puzzle training...")
    print("Training configuration:")
    print(f"  - Max steps: {training_args.max_steps}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Generations per prompt: {training_args.num_generations}")
    print(f"  - Learning rate: {training_args.learning_rate}")

    # Start training
    trainer.train()

    print("Training completed!")
    print(f"Model saved to: {training_args.output_dir}")
