"""What is actually inside a training checkpoint, and under which key are the embeddings?

WHY: the P2 scorer (plan Task 6) has to load an embedding matrix out of a .pth. The plan's test
fixture assumed the key `embeddings`, with an instruction to "verify it". Guessing a key name is how
the Task 8/9 cross-check ended up comparing three fields, one of which was the join key -- a check
that could not fail. So establish the schema from a REAL checkpoint before anyone writes the loader.

Also reports how the scorer already in the repo (scripts/score_recipe_checkpoints.py) gets its
embeddings, so the new scorer matches an existing, working path rather than inventing one.

Read-only. Usage: <venv-python> helpers/p2_inspect_checkpoint_schema.py <checkpoint.pth>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def describe(v, indent: int = 4) -> str:
    pad = " " * indent
    if torch.is_tensor(v):
        return f"{pad}Tensor shape={tuple(v.shape)} dtype={v.dtype}"
    if isinstance(v, dict):
        inner = "\n".join(
            f"{pad}  {k!r}: " + describe(x, 0).strip() for k, x in list(v.items())[:12]
        )
        return f"{pad}dict({len(v)} keys)\n{inner}"
    if isinstance(v, (list, tuple)):
        return f"{pad}{type(v).__name__} len={len(v)}"
    return f"{pad}{type(v).__name__} = {v!r}"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    obj = torch.load(path, map_location="cpu", weights_only=False)

    print(f"checkpoint: {path.name}")
    print(f"top-level type: {type(obj).__name__}")
    if not isinstance(obj, dict):
        print(describe(obj))
        return 0

    print(f"top-level keys ({len(obj)}): {sorted(obj)}\n")
    for k, v in obj.items():
        print(f"  {k!r}:")
        print(describe(v))

    # which key holds a 2-D float matrix that could be the embedding table?
    print("\ncandidate embedding tables (2-D float tensors):")
    found = []
    for k, v in obj.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.is_floating_point():
            found.append((k, tuple(v.shape)))
            print(f"  TOP-LEVEL {k!r} shape={tuple(v.shape)}")
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if torch.is_tensor(v2) and v2.dim() == 2 and v2.is_floating_point():
                    found.append((f"{k}.{k2}", tuple(v2.shape)))
                    print(f"  NESTED   {k!r}.{k2!r} shape={tuple(v2.shape)}")
    if not found:
        print("  NONE -- the loader cannot assume a 2-D float tensor is present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
