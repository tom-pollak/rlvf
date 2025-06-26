"""
Dataset loading and processing for culture puzzles using HuggingFace datasets.

This module provides clean interfaces for loading, processing, and transforming
the culture puzzles dataset using the HuggingFace datasets API.
"""

from functools import partial
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

from .grids import decode_puzzle_grids
from .prompts import PuzzleOrder, PuzzlePromptGenerator


def format_puzzle_batch(
    batch: dict[str, list], generator: PuzzlePromptGenerator
) -> dict[str, list]:
    questions = []
    answers = []
    grids = []

    for input_ids in batch["input_ids"]:
        grid_dict = decode_puzzle_grids(input_ids)
        question, answer = generator.create_prompt(grid_dict)
        grids.append(grid_dict)
        questions.append(question)
        answers.append(answer)

    return {"question": questions, "answer": answers, "grids": grids}


def mk_dataset_single(
    order: PuzzleOrder,
    num_samples: Optional[int] = None,
    batch_size: int = 1000,
    num_proc: Optional[int] = None,
) -> tuple[Dataset, Dataset | None]:
    """
    Create train and test datasets for culture puzzles.

    Args:
        num_samples: Optional limit on number of samples to load
        test_size: Fraction of data to use for testing
        order: Puzzle ordering strategy
        batch_size: Batch size for processing
        num_proc: Number of processes for parallel processing

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    dataset = load_dataset("tommyp111/culture-puzzles-1M", split="train")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    generator = PuzzlePromptGenerator(order)
    dataset = dataset.map(
        partial(format_puzzle_batch, generator=generator),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc=f"Processing culture puzzles ({order.value})",
        remove_columns=dataset.column_names,
    )
    return dataset


def mk_dataset(**kwargs):
    orderings = [
        PuzzleOrder.FORWARD,
        PuzzleOrder.REVERSE,
        PuzzleOrder.ALT_FORWARD,
        PuzzleOrder.ALT_REVERSE,
    ]

    return DatasetDict(
        {order.value: mk_dataset_single(order, **kwargs) for order in orderings}
    )
