import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from taxembed.utils.training_pairs import TrainingPairs

REPO = Path(__file__).resolve().parents[2]
MOLLUSCA = REPO / "data/taxopy/mollusca_6447_clean/taxonomy_edges_mollusca_6447_clean_transitive.npz"


@pytest.mark.skipif(not MOLLUSCA.exists(), reason="mollusca closure not on this machine")
def test_builder_writes_a_loadable_split_and_a_manifest_that_adds_up(tmp_path):
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_p2_split.py"),
         "--npz", str(MOLLUSCA), "--outdir", str(tmp_path),
         "--visibility", "0.5", "--seed", "0", "--frac-test", "0.10", "--frac-val", "0.05"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr

    manifests = list(tmp_path.glob("*_manifest.json"))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text())

    train = TrainingPairs.load(tmp_path / m["train_npz"])
    held = np.load(tmp_path / m["heldout_npz"])

    # the manifest's arithmetic must close
    assert len(train) == m["n_pairs_train"]
    assert m["n_pairs_source"] == (
        m["n_pairs_train"] + m["n_pairs_removed_parent_edges"] + m["n_pairs_removed_visibility"]
    )
    # test and val are disjoint
    assert len(np.intersect1d(held["test"], held["val"])) == 0
    # NO held-out node has a depth_diff==1 row left anywhere in the training file
    held_all = np.concatenate([held["test"], held["val"]])
    dd1_children = set(train.descendant_idx[train.depth_diff == 1].tolist())
    assert dd1_children.isdisjoint(set(held_all.tolist()))
    # every held-out node still appears (it needs a coordinate)
    appearing = set(train.descendant_idx.tolist()) | set(train.ancestor_idx.tolist())
    assert set(held_all.tolist()).issubset(appearing)


@pytest.mark.skipif(not MOLLUSCA.exists(), reason="mollusca closure not on this machine")
def test_visibility_zero_keeps_only_parent_edges(tmp_path):
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_p2_split.py"),
         "--npz", str(MOLLUSCA), "--outdir", str(tmp_path),
         "--visibility", "0.0", "--seed", "0", "--frac-test", "0.10", "--frac-val", "0.05"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    m = json.loads(next(tmp_path.glob("*_manifest.json")).read_text())
    train = TrainingPairs.load(tmp_path / m["train_npz"])
    assert (train.depth_diff == 1).all()


@pytest.mark.skipif(not MOLLUSCA.exists(), reason="mollusca closure not on this machine")
def test_same_seed_reproduces_the_same_split_md5(tmp_path):
    outs = []
    for sub in ("a", "b"):
        d = tmp_path / sub
        d.mkdir()
        subprocess.run(
            [sys.executable, str(REPO / "scripts/build_p2_split.py"),
             "--npz", str(MOLLUSCA), "--outdir", str(d),
             "--visibility", "0.5", "--seed", "7", "--frac-test", "0.10", "--frac-val", "0.05"],
            capture_output=True, text=True, check=True,
        )
        outs.append(json.loads(next(d.glob("*_manifest.json")).read_text())["heldout_md5"])
    assert outs[0] == outs[1]
