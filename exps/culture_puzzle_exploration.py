# %%

import os
import sys

# Add the src directory to the path so we can import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlvf.culture_puzzles import PuzzleOrder, mk_dataset

# %%

orderings = [
    PuzzleOrder.FORWARD,
    PuzzleOrder.REVERSE,
    PuzzleOrder.ALT_FORWARD,
    PuzzleOrder.ALT_REVERSE,
]

order_dd = mk_dataset()
order_dd.push_to_hub("tommyp111/culture-puzzles-1M-prompt")
#
# for order, ds in order_dd.items():
#     print(f"\n{order.upper()} ORDER:")
#
#     print(ds[0]["question"])
#     print(ds[0]["answer"])
#     print(ds[0]["grids"])
#     print(ds.features)
#     print("-" * 30)
#
