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
def test_visibility_zero_keeps_parent_edges_plus_heldout_ancestry(tmp_path):
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_p2_split.py"),
         "--npz", str(MOLLUSCA), "--outdir", str(tmp_path),
         "--visibility", "0.0", "--seed", "0", "--frac-test", "0.10", "--frac-val", "0.05"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    m = json.loads(next(tmp_path.glob("*_manifest.json")).read_text())
    train = TrainingPairs.load(tmp_path / m["train_npz"])
    held = np.load(tmp_path / m["heldout_npz"])
    held_all = set(np.concatenate([held["test"], held["val"]]).tolist())
    # At visibility 0.0 the training file holds the retained nodes' parent edges
    # (depth_diff == 1) PLUS the held-out nodes' dd>=2 ancestry rows -- exempted from
    # thinning so every held-out node keeps at least one trainable coordinate anchor.
    deep = train.depth_diff >= 2
    n_deep = int(deep.sum())
    deep_nodes = set(train.descendant_idx[deep].tolist())

    # 🛑 THIS TEST USED TO BE VACUOUS, and it was named for the fix it failed to protect.
    # The old assertion was `set(descendant_idx[deep]).issubset(held_all)`. Delete the
    # exemption (`& ~is_heldout_row` in build_p2_split.py) and NO dd>=2 row survives at
    # visibility 0.0, so `deep_nodes` is empty -- and the empty set is a subset of
    # anything, so the test passed with the Task-2 plan-defect fix removed. Reproduced in
    # a shadow tree: `helpers/p2_lesion_harness.py --mutation heldout_exemption`.
    # The three assertions below each fail on that lesion.

    # 1. The exemption actually kept rows -- this is what kills the empty-set pass.
    assert n_deep > 0, "no dd>=2 rows survived: the held-out exemption is not doing anything"
    # 2. It kept exactly the number the manifest claims (ties the artifact to its bookkeeping).
    assert n_deep == m["n_pairs_heldout_ancestry_kept"]
    # 3. EQUALITY, not issubset. `issubset` is blind to UNDER-application (the empty set);
    #    equality also catches OVER-application, i.e. an exemption that leaked retained
    #    nodes' deep rows into the training file.
    assert deep_nodes == held_all

    # 4. Each held-out node keeps its FULL ancestry above the dropped parent edge: a node at
    #    depth d has ancestors at depth_diff 2..d, so exactly d-1 rows. This is the identity
    #    hand-verified when the exemption was introduced (mollusca 7,693 / 740 = 10.4 rows
    #    per node, mean depth ~11.4 against a max depth of 14). It pins the exemption's
    #    EXTENT, not merely its existence -- a partial exemption passes 1-3 but fails here.
    counts = np.bincount(train.descendant_idx[deep], minlength=train.n_nodes)
    depth_of = np.zeros(train.n_nodes, dtype=np.int64)
    depth_of[train.descendant_idx] = train.descendant_depth
    for v in sorted(deep_nodes):
        assert counts[v] == depth_of[v] - 1, (
            f"held-out node {v} at depth {depth_of[v]} kept {counts[v]} dd>=2 rows, expected "
            f"{depth_of[v] - 1}"
        )


@pytest.mark.skipif(not MOLLUSCA.exists(), reason="mollusca closure not on this machine")
@pytest.mark.parametrize("visibility", [0.0, 0.5])
def test_every_heldout_node_has_at_least_one_training_row(tmp_path, visibility):
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_p2_split.py"),
         "--npz", str(MOLLUSCA), "--outdir", str(tmp_path),
         "--visibility", str(visibility), "--seed", "0", "--frac-test", "0.10", "--frac-val", "0.05"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    m = json.loads(next(tmp_path.glob("*_manifest.json")).read_text())
    train = TrainingPairs.load(tmp_path / m["train_npz"])
    held = np.load(tmp_path / m["heldout_npz"])
    held_all = set(np.concatenate([held["test"], held["val"]]).tolist())
    # every held-out node needs a coordinate: it must appear in at least one training row
    appearing = set(train.descendant_idx.tolist()) | set(train.ancestor_idx.tolist())
    assert held_all.issubset(appearing)


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


def _write_synthetic_closure(path: Path):
    """0 -> {1,2}; 1 -> {3,4}; 2 -> {5,6} -- full transitive closure (direct + depth_diff=2 rows),
    saved as a TrainingPairs .npz. Independent of the Mollusca data file (not on every machine),
    so the --frac-val default test below always runs."""
    direct = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]      # depth_diff == 1
    grandparent = [(0, 3), (0, 4), (0, 5), (0, 6)]                 # depth_diff == 2
    depth = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2}
    rows = [(a, d, 1) for a, d in direct] + [(a, d, 2) for a, d in grandparent]
    n = len(rows)
    pairs = TrainingPairs(
        ancestor_idx=np.array([a for a, d, dd in rows], dtype=np.int32),
        descendant_idx=np.array([d for a, d, dd in rows], dtype=np.int32),
        depth_diff=np.array([dd for a, d, dd in rows], dtype=np.int16),
        ancestor_depth=np.array([depth[a] for a, d, dd in rows], dtype=np.int16),
        descendant_depth=np.array([depth[d] for a, d, dd in rows], dtype=np.int16),
        ancestor_taxid=np.array([a for a, d, dd in rows], dtype=np.int32),
        descendant_taxid=np.array([d for a, d, dd in rows], dtype=np.int32),
    )
    assert len(pairs) == n
    pairs.save(path)


def test_frac_val_default_is_zero_and_the_flag_still_works_when_passed(tmp_path):
    """IMPORTANT #3 (fix round 1): the old --frac-val default of 0.05 silently withheld 5% of
    eligible nodes from every score_p2_linkpred.py number whenever the flag was forgotten. The
    default must now be 0.0 (nothing withheld without an explicit flag), and passing --frac-val
    explicitly must still work exactly as before. Uses a synthetic closure, not Mollusca, so it
    is not gated on data availability."""
    npz = tmp_path / "synthetic_transitive.npz"
    _write_synthetic_closure(npz)

    default_dir = tmp_path / "default"
    default_dir.mkdir()
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_p2_split.py"),
         "--npz", str(npz), "--outdir", str(default_dir),
         "--visibility", "0.5", "--seed", "0", "--frac-test", "0.5", "--band", "2", "2"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    m_default = json.loads(next(default_dir.glob("*_manifest.json")).read_text())
    assert m_default["n_val"] == 0
    held_default = np.load(default_dir / m_default["heldout_npz"])
    assert len(held_default["val"]) == 0

    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_p2_split.py"),
         "--npz", str(npz), "--outdir", str(explicit_dir),
         "--visibility", "0.5", "--seed", "0", "--frac-test", "0.5", "--frac-val", "0.5",
         "--band", "2", "2"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    m_explicit = json.loads(next(explicit_dir.glob("*_manifest.json")).read_text())
    assert m_explicit["n_val"] == 2  # int(4 eligible leaves * 0.5), the flag still takes effect
    held_explicit = np.load(explicit_dir / m_explicit["heldout_npz"])
    assert len(held_explicit["val"]) == 2
