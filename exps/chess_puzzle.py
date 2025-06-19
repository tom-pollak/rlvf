"""
Chess Puzzle Training with GRPO

This script trains a language model to solve chess puzzles from the Lichess dataset
using Group Relative Policy Optimization (GRPO) with the verifiers framework.

Usage:
# Start vLLM inference server
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model 'Qwen/Qwen2.5-1.5B-Instruct' --tensor-parallel-size 2

# Run training
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file verifiers/configs/zero3.yaml exps/chess_puzzle.py
"""

import verifiers as vf
from datasets import load_dataset

size = "1.5B"
model_name = f"Qwen/Qwen2.5-{size}-Instruct"

## Dataset

dataset = load_dataset("Lichess/chess-puzzles", split="train")


def format_chess_puzzle(example):
    """Format chess puzzle for training"""
    return {
        "question": f"FEN: {example['FEN']}\nSolve this chess puzzle.",
        "answer": example["Moves"],
    }


dataset = dataset.map(format_chess_puzzle)

eval_size = min(1000, len(dataset) // 10)  # 10% or 1000 examples for eval
eval_dataset = dataset.select(range(eval_size))
train_dataset = dataset.select(range(eval_size, len(dataset)))

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

## Environment
parser = vf.XMLParser(["think", "answer"])

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


def chess_moves_reward(completion, answer, **kwargs):
    """
    Reward function that gives 1.0 for exact match, 0.0 otherwise.
    Could be enhanced with partial credit for correct first moves.
    """
    predicted = parser.parse_answer(completion) or ""
    predicted = predicted.strip()
    answer = answer.strip()

    # # Exact match
    # if predicted == answer:
    #     return 1.0

    # Correct first move
    if predicted and answer:
        pred_moves = predicted.split()
        answer_moves = answer.split()
        if len(pred_moves) > 0 and len(answer_moves) > 0:
            if pred_moves[0] == answer_moves[0]:
                return 1.0

    return 0.0


rubric = vf.Rubric(
    funcs=[chess_moves_reward, parser.get_format_reward_func()],
    weights=[1.0, 0.2],  # (correct move, format)
)

env = vf.SingleTurnEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)

## Model and Trainer
model, tokenizer = vf.get_model_and_tokenizer(model_name)

training_args = vf.grpo_defaults(run_name=f"chess-puzzle-grpo-{size}")
training_args.per_device_train_batch_size = 6
training_args.num_generations = 12
training_args.gradient_accumulation_steps = 4
training_args.max_prompt_length = 512
training_args.max_completion_length = 4096
training_args.max_steps = 1000
training_args.eval_steps = 100
training_args.save_steps = 200
training_args.logging_steps = 10

trainer = vf.GRPOTrainer(
    model=model, processing_class=tokenizer, env=env, args=training_args
)

## Train!
trainer.train()
