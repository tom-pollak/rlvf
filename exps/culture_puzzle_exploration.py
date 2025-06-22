# %%
"""
Culture Puzzles Data Exploration

This script explores the tommyp111/culture-puzzles-1M dataset and provides
helper functions to visualize the grid-based puzzles in ASCII format.

The dataset format:
- Each sample has 405 integers in input_ids
- First integer (0) is conditioning token
- Next 404 integers contain 4 grids marked by 11, 12, 13, 14
- Each grid is 10x10 = 100 integers after its marker
"""

import torch
import numpy as np
from datasets import load_dataset
from typing import List, Tuple, Dict, Optional

# %%
# Load the dataset
print("Loading culture puzzles dataset...")
dataset = load_dataset("tommyp111/culture-puzzles-1M", split="train")
print(f"Dataset loaded: {len(dataset)} samples")

# %%
# Examine the first few samples
print("First 3 samples structure:")
for i in range(3):
    sample = dataset[i]
    input_ids = sample["input_ids"]
    print(f"Sample {i}: input_ids length = {len(input_ids)}")
    print(f"First 20 tokens: {input_ids[:20]}")
    print(f"Unique values: {sorted(set(input_ids))}")
    print()


# %%
def decode_puzzle_grids(input_ids: List[int]) -> Dict[str, np.ndarray]:
    """
    Decode the 405-integer format into 4 grids.

    Format:
    - input_ids[0] = 0 (conditioning token)
    - input_ids[1:] contains 4 grids marked by 11, 12, 13, 14
    - Each marker is followed by 100 integers (10x10 grid)

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


# %%
def grid_to_ascii(grid: np.ndarray, color_map: Optional[Dict[int, str]] = None) -> str:
    """
    Convert a 10x10 integer grid to ASCII representation.

    Args:
        grid: 10x10 numpy array of integers
        color_map: Optional mapping from integers to characters

    Returns:
        String representation of the grid
    """
    if color_map is None:
        # Default mapping: 0=' ', 1-10='A'-'J'
        color_map = {0: "."}
        for i in range(1, 11):
            color_map[i] = chr(ord("A") + i - 1)

    lines = []
    for row in grid:
        line = " ".join(color_map.get(val, "?") for val in row)
        lines.append(line)

    return "\n".join(lines)


# %%
def visualize_puzzle(input_ids: List[int], sample_idx: int = 0) -> None:
    """
    Visualize a complete puzzle with all 4 grids in ASCII format.
    """
    grids = decode_puzzle_grids(input_ids)

    print(f"=== Puzzle Sample {sample_idx} ===")
    print(f"Found {len(grids)} grids with markers: {list(grids.keys())}")
    print()

    # Map markers to likely grid names based on order
    marker_order = sorted(grids.keys())
    grid_names = ["A", "f_A", "B", "f_B"]

    for i, marker in enumerate(marker_order):
        if i < len(grid_names):
            grid_name = grid_names[i]
        else:
            grid_name = f"Grid_{marker}"

        print(f"Grid {grid_name} (marker {marker}):")
        print(grid_to_ascii(grids[marker]))
        print()


# %%
def analyze_grid_structure(input_ids: List[int]) -> Dict:
    """
    Analyze the structure of a puzzle to understand grid relationships.
    """
    grids = decode_puzzle_grids(input_ids)
    analysis = {
        "num_grids": len(grids),
        "markers_found": list(grids.keys()),
        "grid_shapes": {marker: grid.shape for marker, grid in grids.items()},
        "unique_values": {},
        "non_zero_cells": {},
    }

    for marker, grid in grids.items():
        unique_vals = np.unique(grid)
        non_zero_count = np.count_nonzero(grid)

        analysis["unique_values"][marker] = unique_vals.tolist()
        analysis["non_zero_cells"][marker] = int(non_zero_count)

    return analysis


# %%
# Analyze first 10 samples
print("=== Dataset Analysis ===")
for i in range(10):
    sample = dataset[i]
    analysis = analyze_grid_structure(sample["input_ids"])
    print(
        f"Sample {i}: {analysis['num_grids']} grids, markers {analysis['markers_found']}"
    )

# %%
# Visualize first few puzzles
print("\n=== Puzzle Visualizations ===")
for i in range(3):
    sample = dataset[i]
    visualize_puzzle(sample["input_ids"], i)
    print("-" * 50)


# %%
def understand_transformations(input_ids: List[int]) -> None:
    """
    Try to understand what transformation is being applied between grids.
    """
    grids = decode_puzzle_grids(input_ids)

    if len(grids) >= 4:
        markers = sorted(grids.keys())

        print("Analyzing potential transformations:")

        # Assume first two grids are A -> f_A, next two are B -> f_B
        if len(markers) >= 4:
            A, f_A, B, f_B = [grids[m] for m in markers[:4]]

            # Compare A vs f_A
            a_diff = np.sum(A != f_A)
            print(f"A vs f_A: {a_diff} cells differ")

            # Compare B vs f_B
            b_diff = np.sum(B != f_B)
            print(f"B vs f_B: {b_diff} cells differ")

            # Check if transformation is similar
            if a_diff > 0 and b_diff > 0:
                print("Both pairs show transformations")

                # Look for patterns
                a_changes = np.where(A != f_A)
                b_changes = np.where(B != f_B)

                print(f"A transformation affects {len(a_changes[0])} positions")
                print(f"B transformation affects {len(b_changes[0])} positions")


# %%
# Analyze transformations in first few samples
print("\n=== Transformation Analysis ===")
for i in range(5):
    print(f"\nSample {i}:")
    sample = dataset[i]
    understand_transformations(sample["input_ids"])


# %%
def create_llm_prompt(
    input_ids: List[int], mask_last_grid: bool = True
) -> Tuple[str, str]:
    """
    Create a prompt suitable for LLM training.

    Args:
        input_ids: Raw puzzle data
        mask_last_grid: If True, mask the last grid as the target to predict

    Returns:
        Tuple of (prompt, target_answer)
    """
    grids = decode_puzzle_grids(input_ids)

    if len(grids) < 4:
        raise ValueError(f"Expected 4 grids, got {len(grids)}")

    markers = sorted(grids.keys())

    # Assume order is A, f_A, B, f_B
    A = grids[markers[0]]
    f_A = grids[markers[1]]
    B = grids[markers[2]]
    f_B = grids[markers[3]]

    prompt = f"""You are solving a grid transformation puzzle. Given a pattern A → f_A, apply the same transformation to B to get f_B.

Grid A:
{grid_to_ascii(A)}

Grid f_A (transformation of A):
{grid_to_ascii(f_A)}

Grid B:
{grid_to_ascii(B)}

Grid f_B (apply the same transformation to B):"""

    target = grid_to_ascii(f_B)

    return prompt, target


# %%
# Test prompt creation
print("\n=== LLM Prompt Example ===")
sample = dataset[0]
prompt, target = create_llm_prompt(sample["input_ids"])
print(prompt)
print("\nTarget answer:")
print(target)

print("\nPrompt + Target combined:")
print(prompt + "\n" + target)

# %%
