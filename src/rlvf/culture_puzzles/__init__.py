# Dataset utilities
from .dataset import mk_dataset

# Grid utilities
from .grids import (
    ascii_to_grid,
    grid_to_ascii,
)

# Prompt utilities
from .prompts import (
    PuzzleOrder,
    PuzzlePromptGenerator,
    create_puzzle_prompt,
)

# Version info
__version__ = "0.1.0"

# Public API
__all__ = [
    # Core classes
    "PuzzleOrder",
    "PuzzlePromptGenerator",
    # Grid functions
    "grid_to_ascii",
    "ascii_to_grid",
    # Prompt functions
    "create_puzzle_prompt",
    "mk_dataset",
]
