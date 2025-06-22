# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLVF is a reinforcement learning framework for training language models with verifiable rewards, specifically designed for the chess puzzle training task. The project uses the `verifiers` library for GRPO (Group Relative Policy Optimization) training.

The main goal is to finetune a model on chess puzzles from the Lichess dataset, where the model must predict the exact sequence of moves to solve each puzzle.

## Core Environment Types

The verifiers library provides several environment types for different RL training scenarios:

### 1. SingleTurnEnv - Simple Q&A Tasks

Perfect for tasks like math problems, text completion, or chess puzzles where you need one response.

```python
import verifiers as vf

# Chess puzzle example
parser = vf.XMLParser(['think', 'answer'])
def chess_reward(completion, answer, **kwargs):
    predicted_moves = parser.parse_answer(completion)
    return 1.0 if predicted_moves == answer else 0.0

env = vf.SingleTurnEnv(
    dataset=chess_dataset,  # HF Dataset with 'question'+'answer'
    system_prompt=f"Solve the chess puzzle. {parser.get_format_str()}",
    parser=parser,
    rubric=vf.Rubric([chess_reward], weights=[1.0])
)
```

**Reference**: See `verifiers/examples/gsm8k.py` for math problems, `verifiers/examples/reverse_text.py` for text tasks.

### 2. ToolEnv - Multi-turn Tool Usage

For agents that need to use tools like Python execution, search, or calculators.

```python
from verifiers.tools import python

env = vf.ToolEnv(
    dataset=dataset,
    tools=[python],
    max_turns=10,
    system_prompt="""Use tools to solve the problem step by step.
    Call tools with <tool>{"name": "python", "args": {"code": "..."}}</tool>
    End with <answer>final answer</answer>""",
    rubric=vf.ToolRubric([reward_func])
)
```

**Reference**: See `verifiers/examples/math_python.py` for complete tool usage patterns.

### 3. TextArenaEnv - Game Environments

For interactive game-like environments (Wordle, etc.).

```python
env = vf.TextArenaEnv(
    game="Wordle-v0",
    num_samples=1000,
    max_concurrent=32
)
```

**Reference**: See `verifiers/examples/wordle.py` for game environment setup.

### 4. Custom MultiTurnEnv

For implementing custom interaction protocols.

```python
class ChessEnv(vf.MultiTurnEnv):
    def is_completed(self, messages, state, **kwargs):
        # Return True when puzzle is solved or max turns reached
        return state.get('solved', False) or len(messages) > 10

    def env_response(self, messages, state, **kwargs):
        # Process move and return board state
        last_move = messages[-1]['content']
        new_state = self.process_move(last_move, state)
        response = {"role": "assistant", "content": f"Board: {new_state['board']}"}
        return response, new_state
```

## Training Patterns

### Basic Training Script

```python
import verifiers as vf

# 1. Load model
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

# 2. Create environment
env = vf.SingleTurnEnv(...)

# 3. Configure training
training_args = vf.grpo_defaults(run_name="chess-puzzle-training")
training_args.max_steps = 500
training_args.per_device_train_batch_size = 4
training_args.num_generations = 8

# 4. Train
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=env, args=training_args)
trainer.train()
```

### Chess Puzzle Training

For the chess puzzle task, see the complete implementation in `chess_puzzle_experiment.py`.

## Running Training

### Start vLLM Server

```bash
# Single GPU inference
CUDA_VISIBLE_DEVICES=0 vf-vllm --model 'Qwen/Qwen2.5-1.5B-Instruct'

# Multi-GPU inference
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model 'Qwen/Qwen2.5-1.5B-Instruct' --tensor-parallel-size 2
```

### Run GRPO Training

```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml train.py

# Multi-GPU training
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 --config-file configs/zero3.yaml train.py
```

## Key Examples to Reference

- **`verifiers/examples/gsm8k.py`** - Math problem solving with exact answer matching
- **`verifiers/examples/reverse_text.py`** - Text manipulation with LCS scoring
- **`verifiers/examples/math_python.py`** - Multi-turn tool usage with Python execution
- **`verifiers/examples/wordle.py`** - Game environment with TextArena
- **`verifiers/examples/sft/`** - SFT warmup scripts for cold-start training

## Common Reward Functions

```python
# Exact match reward
def exact_match_reward(completion, answer, **kwargs):
    predicted = parser.parse_answer(completion) or ''
    return 1.0 if predicted.strip() == answer.strip() else 0.0

# Substring reward
def contains_reward(completion, answer, **kwargs):
    predicted = parser.parse_answer(completion) or ''
    return 1.0 if answer in predicted else 0.0

# Similarity reward using LCS
def lcs_reward(completion, answer, **kwargs):
    from difflib import SequenceMatcher
    predicted = parser.parse_answer(completion) or ''
    return SequenceMatcher(None, predicted, answer).ratio()
```

## Important Notes

- Use `SingleTurnEnv` for the chess puzzle task - it's perfect for one-shot puzzle solving
- All environments support async batch processing with configurable concurrency
- Set `NCCL_P2P_DISABLE=1` if experiencing GPU communication issues
- The framework requires vLLM server for efficient generation during training

## Interpretability Culture Repository

The `interpretability-culture/culture/` submodule contains an experimental framework by François Fleuret exploring emergent cognition through competitive language model training.

### Core Concept

The project tests the hypothesis that high-level cognition emerges from competition among agents in language/idea space. Communicating agents try to create content that is "smart but doable" - understandable by some agents but not all.

### Key Components

- **World Models**: 6x8 grids with moving "birds" (objects with color/direction)
- **Competitive Training**: 5 GPT models compete by creating and solving quizzes
- **Quiz Generation**: Models create puzzles that are solvable by most but not all competitors
- **Culture Evolution**: Successful quizzes are added to shared training set

### Training Process

1. Train models on world physics (bird movement in grids)
2. Select worst-performing model for improvement
3. When model achieves >97.5% accuracy, it generates new quizzes
4. Valid quizzes: solvable by all but one competitor in both time directions
5. Add 1000 validated quizzes to "culture" (shared training set)
6. Repeat cycle to evolve increasingly complex concepts

### File Structure

- `main.py` - Main training loop and model management
- `sky.py`, `grids.py`, `wireworld.py`, `life.py` - Different world physics implementations
- `mygpt.py` - Custom GPT implementation with caching
- `quiz_machine.py` - Quiz generation and validation logic
- `problem.py` - Base class for world physics problems

### Expected Emergent Concepts

The framework aims to develop concepts like connectivity, symmetry, interior/exterior, and spatial vocabulary through competitive pressure rather than explicit supervision.
