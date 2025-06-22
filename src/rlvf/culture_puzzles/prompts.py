"""
Prompt generation for culture puzzles with configurable orderings.

This module provides flexible prompt generation that supports different
puzzle presentation orders as used in the culture framework.
"""

from enum import Enum
from typing import List, Tuple

from .grids import decode_puzzle_grids, grid_to_ascii


class PuzzleOrder(Enum):
    """Different orderings for presenting culture puzzles."""

    FORWARD = "forward"  # A, f_A, B, f_B (predict f_B)
    REVERSE = "reverse"  # f_A, A, f_B, B (predict B)
    ALT_FORWARD = "alt_forward"  # B, f_B, A, f_A (predict f_A)
    ALT_REVERSE = "alt_reverse"  # f_B, B, f_A, A (predict A)


class PuzzlePromptGenerator:
    """Generates prompts for culture puzzles with configurable ordering."""

    def __init__(self, order: PuzzleOrder = PuzzleOrder.FORWARD):
        """
        Initialize the prompt generator.

        Args:
            order: The puzzle ordering strategy to use
        """
        self.order = order

        # Define the structure for each ordering
        self._order_configs = {
            PuzzleOrder.FORWARD: {
                "grids": ["A", "f_A", "B", "f_B"],
                "given": ["A", "f_A", "B"],
                "predict": "f_B",
                "pattern": "A → f_A",
                "apply_to": "B",
            },
            PuzzleOrder.REVERSE: {
                "grids": ["f_A", "A", "f_B", "B"],
                "given": ["f_A", "A", "f_B"],
                "predict": "B",
                "pattern": "f_A → A",
                "apply_to": "f_B",
            },
            PuzzleOrder.ALT_FORWARD: {
                "grids": ["B", "f_B", "A", "f_A"],
                "given": ["B", "f_B", "A"],
                "predict": "f_A",
                "pattern": "B → f_B",
                "apply_to": "A",
            },
            PuzzleOrder.ALT_REVERSE: {
                "grids": ["f_B", "B", "f_A", "A"],
                "given": ["f_B", "B", "f_A"],
                "predict": "A",
                "pattern": "f_B → B",
                "apply_to": "f_A",
            },
        }

    def create_prompt(self, grids) -> Tuple[str, str]:
        """
        Create a prompt for the given puzzle with the configured ordering.

        Args:
            input_ids: Raw puzzle data

        Returns:
            Tuple of (question, answer)
        """

        config = self._order_configs[self.order]

        # Build the prompt based on the ordering
        prompt_parts = []

        if self.order == PuzzleOrder.FORWARD:
            prompt_parts.append(
                "You are solving a grid transformation puzzle. Study the pattern A → f_A, then apply the same transformation to grid B."
            )
        elif self.order == PuzzleOrder.REVERSE:
            prompt_parts.append(
                "You are solving a reverse grid transformation puzzle. Study the pattern f_A → A, then apply the same reverse transformation to f_B."
            )
        elif self.order == PuzzleOrder.ALT_FORWARD:
            prompt_parts.append(
                "You are solving a grid transformation puzzle. Study the pattern B → f_B, then apply the same transformation to grid A."
            )
        elif self.order == PuzzleOrder.ALT_REVERSE:
            prompt_parts.append(
                "You are solving a reverse grid transformation puzzle. Study the pattern f_B → B, then apply the same reverse transformation to f_A."
            )

        prompt_parts.append("")

        # Add the given grids
        for grid_name in config["given"]:
            prompt_parts.append(f"Grid {grid_name}:")
            prompt_parts.append(grid_to_ascii(grids[grid_name]))
            prompt_parts.append("")

        # Add the question for the target grid
        prompt_parts.append(
            f"What is Grid {config['predict']} (apply the transformation from {config['pattern']} to {config['apply_to']})?"
        )

        question = "\n".join(prompt_parts)
        answer = grid_to_ascii(grids[config["predict"]])

        return question, answer


def create_puzzle_prompt(
    input_ids: List[int], order: PuzzleOrder = PuzzleOrder.FORWARD
) -> Tuple[str, str]:
    """
    Convenience function to create a puzzle prompt with specified ordering.

    Args:
        input_ids: Raw puzzle data
        order: Puzzle ordering strategy

    Returns:
        Tuple of (question, answer)
    """
    generator = PuzzlePromptGenerator(order)
    return generator.create_prompt(input_ids)
