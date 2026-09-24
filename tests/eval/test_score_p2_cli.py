import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/score_p2_linkpred.py"


def test_scorer_runs_end_to_end_on_a_synthetic_tree(tmp_path):
    """A 3-level tree, an embedding that PERFECTLY encodes it, and one that is noise.
    The perfect arm must score MRR 1.0; the noise arm must land near sibling chance."""
    # 0 -> {1,2}; 1 -> {3,4}; 2 -> {5,6}   (3..6 are leaves at depth 2)
    parent = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    depth = np.array([0, 1, 1, 2, 2, 2, 2], dtype=np.int64)

    held = tmp_path / "heldout.npz"
    np.savez(held, test=np.array([3, 5], dtype=np.int64), val=np.array([], dtype=np.int64))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "clade": "synthetic", "band": [0, 99], "n_nodes": 7,
        "parent": parent.tolist(), "depth": depth.tolist(),
    }))

    # perfect: each leaf sits on top of its true parent
    perfect = np.array([[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0],
                        [0.5, 0.01], [0.5, -0.01], [-0.5, 0.01], [-0.5, -0.01]])
    ck = tmp_path / "perfect_epoch1.pth"
    torch.save({"embeddings": torch.tensor(perfect)}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"perfect={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr

    res = json.loads(out.read_text())
    assert res["arms"]["perfect"]["checkpoints"][0]["metrics"]["mrr"] == 1.0
    assert res["arms"]["perfect"]["checkpoints"][0]["metrics"]["hits_at_1"] == 1.0
    # the mandatory baselines are present and the Vendrov rule finds nothing
    assert res["baselines"]["vendrov_recall"] == 0.0
    assert res["baselines"]["sibling_chance_mean"] == 0.5


def _build_layered_tree(n1: int, n2: int, n3: int):
    """root(0) -> n1 level-1 nodes -> n2 level-2 children each -> n3 level-3 LEAF children each.

    `angle[v]` encodes v's position in the tree as a base-10-ish mixed-radix number, one digit
    per level, with each deeper digit an order of magnitude smaller than the last
    (level1 step 1.0, level2 step 0.1, level3 step 0.01). Two nodes sharing a grandparent but
    different level-2 parents therefore differ in angle by >= 0.1 - 3*0.01 = 0.07, while a leaf
    and its OWN level-2 parent differ by <= 3*0.01 = 0.03 -- a fixed, guaranteed margin, so
    encoding `(cos(angle), sin(angle))` as the embedding makes nearest-neighbour-by-cosine
    recover the true parent for every held-out leaf, deterministically (no near-ties, so the
    scorer's tie-break jitter cannot flip anything).
    """
    parent = [0]
    depth = [0]
    angle = [0.0]
    next_idx = 1

    level1 = []  # (id, c1)
    for c1 in range(n1):
        level1.append((next_idx, c1))
        parent.append(0)
        depth.append(1)
        angle.append(c1 * 1.0)
        next_idx += 1

    level2 = []  # (id, c1, c2)
    for pid, c1 in level1:
        for c2 in range(n2):
            level2.append((next_idx, c1, c2))
            parent.append(pid)
            depth.append(2)
            angle.append(c1 * 1.0 + c2 * 0.1)
            next_idx += 1

    leaves = []
    for pid, c1, c2 in level2:
        for c3 in range(n3):
            leaves.append(next_idx)
            parent.append(pid)
            depth.append(3)
            angle.append(c1 * 1.0 + c2 * 0.1 + c3 * 0.01)
            next_idx += 1

    return (np.array(parent, dtype=np.int64), np.array(depth, dtype=np.int64),
            np.array(angle, dtype=np.float64), np.array(leaves, dtype=np.int64))


def test_scorer_numbers_are_right_perfect_arm_hits_1_scrambled_arm_lands_near_chance(tmp_path):
    """The brief's own fixture is small enough (2 held nodes) that a lucky guess can't be told
    apart from a real signal. Build a bigger tree (131 nodes, 100 leaves, uniform grandparent
    fan-out of 5) so a SCRAMBLED embedding's hit rate is a real statistic against a fixed
    sibling_chance floor, not a coin flip. A scorer that cannot distinguish a perfect model from
    random noise on this fixture is not measuring anything."""
    n1, n2, n3 = 5, 5, 4
    parent, depth, angle, leaves = _build_layered_tree(n1, n2, n3)
    n_nodes = len(parent)
    assert n_nodes == 1 + n1 + n1 * n2 + n1 * n2 * n3
    assert len(leaves) == n1 * n2 * n3

    held_ids = leaves[:40]  # 40 of the 100 leaves, deterministic (not a random subsample)

    heldout_path = tmp_path / "heldout.npz"
    np.savez(heldout_path, test=held_ids, val=np.array([], dtype=np.int64))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "clade": "synthetic_layered", "band": [0, 99], "n_nodes": n_nodes,
        "parent": parent.tolist(), "depth": depth.tolist(),
    }))

    # Every embedding sits on a circle of radius 0.5 (away from the Poincare boundary, so the
    # ball-coordinate poincare fallback stays well-defined too); cosine distance is then a
    # monotonic function of the angular gap alone.
    perfect_emb = 0.5 * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    perfect_ck = tmp_path / "perfect_epoch1.pth"
    torch.save({"embeddings": torch.tensor(perfect_emb)}, perfect_ck)

    # scrambled: same vectors, rows permuted so embedding no longer tracks tree structure at all.
    rng = np.random.default_rng(20260924)
    perm = rng.permutation(n_nodes)
    scrambled_emb = perfect_emb[perm]
    scrambled_ck = tmp_path / "scrambled_epoch1.pth"
    torch.save({"embeddings": torch.tensor(scrambled_emb)}, scrambled_ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--manifest", str(manifest_path), "--heldout", str(heldout_path),
         "--checkpoints", f"perfect={perfect_ck}", "--checkpoints", f"scrambled={scrambled_ck}",
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    res = json.loads(out.read_text())

    # every held leaf's grandparent has exactly n2=5 children -> chance is exactly 1/5 everywhere
    sibling_chance_mean = res["baselines"]["sibling_chance_mean"]
    assert sibling_chance_mean == pytest.approx(1.0 / n2)

    perfect_metrics = res["arms"]["perfect"]["checkpoints"][0]["metrics"]
    assert perfect_metrics["mrr"] == 1.0
    assert perfect_metrics["hits_at_1"] == 1.0
    assert perfect_metrics["n"] == len(held_ids)

    scrambled_metrics = res["arms"]["scrambled"]["checkpoints"][0]["metrics"]
    # far worse than the perfect model, and within a generous band of the chance floor -- a
    # scrambled embedding carries no structure, so it must not look anywhere near "learned".
    assert scrambled_metrics["mrr"] < 0.7
    assert abs(scrambled_metrics["hits_at_1"] - sibling_chance_mean) < 0.15
