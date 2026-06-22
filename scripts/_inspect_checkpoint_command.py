#!/usr/bin/env python
"""Dump the non-tensor metadata (saved command / config / args) from a .pth checkpoint.

Read-only diagnostic: prints the training command and config a checkpoint was
created with, so an experiment can be reproduced with a single flag changed.
Does not modify anything.

Usage:
    .venv/bin/python scripts/_inspect_checkpoint_command.py <checkpoint.pth>
"""
import sys
import json

import torch


def jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


def main():
    path = sys.argv[1]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print(f"checkpoint is a {type(ckpt)}, not a dict")
        return
    print("== top-level keys ==")
    for k, v in ckpt.items():
        kind = type(v).__name__
        if torch.is_tensor(v):
            print(f"  {k}: Tensor{tuple(v.shape)}")
        else:
            print(f"  {k}: {kind}")
    print()
    for key in ("command", "cmd", "args", "config", "metadata", "run", "hparams"):
        if key in ckpt:
            print(f"== {key} ==")
            print(json.dumps(jsonable(ckpt[key]), indent=2, default=str))
            print()


if __name__ == "__main__":
    main()
