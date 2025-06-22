"""
Grid handling utilities for culture puzzles.

This module provides functions for decoding, encoding, and visualizing
10x10 grid puzzles from the culture dataset.
"""

from typing import Dict, Optional

import numpy as np

markers = {11: "A", 12: "f_A", 13: "B", 14: "f_B"}


def decode_puzzle_grids(input_ids) -> Dict[int, np.ndarray]:
    """
    Decode the 405-integer format into 4 grids.

    Format:
    - input_ids[0] = 0 (conditioning token)
    - input_ids[1:] contains 4 grids marked by 11, 12, 13, 14
    - Each marker is followed by 100 integers (10x10 grid)

    Args:
        input_ids: list of 405 integers representing the puzzle

    Returns:
        Dict mapping marker positions to 10x10 numpy arrays
    """
    input_ids = np.array(input_ids)

    # Skip the conditioning token
    tokens = input_ids[1:]

    # Find marker positions
    grids = {}
    for marker_val in [11, 12, 13, 14]:
        positions = np.where(tokens == marker_val)[0]
        if len(positions) == 0:
            continue
        assert len(positions) == 1

        marker_name = markers[marker_val]
        start_idx = positions[0] + 1
        end_idx = start_idx + 100

        if end_idx <= len(tokens):
            grid_data = tokens[start_idx:end_idx]
            grid = grid_data.reshape(10, 10)
            grids[marker_name] = grid

    return grids


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

    Args:
        ascii_str: ASCII string representation of grid
        color_map: Optional mapping from characters to integers

    Returns:
        10x10 numpy array or None if parsing fails
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


