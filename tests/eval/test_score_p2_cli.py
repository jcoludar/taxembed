import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/score_p2_linkpred.py"

sys.path.insert(0, str(REPO / "scripts"))
from score_p2_linkpred import (  # noqa: E402
    _vendrov_recall_core, _visible_basic_edges, vendrov_recall,
)
from taxembed.eval import baselines  # noqa: E402
from taxembed.eval.preregistration import (  # noqa: E402
    P2_ROLL_WINDOW, merge_p2_scorer_outputs, p2_verdict,
)
from taxembed.eval.randomdag import closure_from_parent  # noqa: E402


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


def _write_small_tree(tmp_path, test=(3, 5), val=()):
    """0 -> {1,2}; 1 -> {3,4}; 2 -> {5,6} -- the module's canonical small fixture tree, factored
    out here so the radius-guard and val-handling tests below don't each re-derive it."""
    parent = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    depth = np.array([0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    held = tmp_path / "heldout.npz"
    np.savez(held, test=np.array(test, dtype=np.int64), val=np.array(val, dtype=np.int64))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "clade": "synthetic", "band": [0, 99], "n_nodes": 7,
        "parent": parent.tolist(), "depth": depth.tolist(),
    }))
    return manifest, held


_SMALL_TREE_BALL_COORDS = torch.tensor([
    [0.0, 0.0], [0.5, 0.0], [-0.5, 0.0],
    [0.5, 0.01], [0.5, -0.01], [-0.5, 0.01], [-0.5, -0.01],
])


# --- Important 1 (fix round 1): the radius guard must match the checkpoint's actual path ---


def test_radius_guard_fires_on_z_embeddings_path_when_a_z_norm_exceeds_100(tmp_path):
    """|z| is genuinely unbounded on the z_embeddings path, so a synthetic checkpoint with one
    node's |z| > 100 must trip radius_overflow_risk under the "z_norm_gt_100" rule."""
    manifest, held = _write_small_tree(tmp_path)
    z_over = torch.zeros(7, 2, dtype=torch.float64)
    z_over[0] = torch.tensor([150.0, 0.0], dtype=torch.float64)  # one node's |z| = 150 > 100
    ck = tmp_path / "zover_epoch1.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS, "z_embeddings": z_over}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"zover={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    row = json.loads(out.read_text())["arms"]["zover"]["checkpoints"][0]
    assert row["poincare_radius_source"] == "z_embeddings"
    assert row["radius_guard"] == "z_norm_gt_100"
    assert row["radius_overflow_risk"] is True
    assert row["max_radius"] == pytest.approx(150.0)
    assert row["n_norm_saturated"] is None
    assert "exceeds" in rc.stdout


def test_radius_guard_silent_on_z_embeddings_path_when_all_z_norms_are_small(tmp_path):
    """The mirror case: every |z| comfortably under 100 must NOT trip the guard."""
    manifest, held = _write_small_tree(tmp_path)
    z_under = torch.full((7, 2), 1.0, dtype=torch.float64)  # |z| = sqrt(2) =~ 1.41
    ck = tmp_path / "zunder_epoch1.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS, "z_embeddings": z_under}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"zunder={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    row = json.loads(out.read_text())["arms"]["zunder"]["checkpoints"][0]
    assert row["radius_guard"] == "z_norm_gt_100"
    assert row["radius_overflow_risk"] is False


def test_radius_guard_fires_on_ball_coordinates_path_when_a_norm_saturates(tmp_path):
    """Without z_embeddings, radius must be derived from the (float32) ball-coordinate norm,
    which can never reach the old >100 threshold (float32 cannot represent anything closer to 1.0
    than ~1.19e-7). A synthetic checkpoint with one row at norm >= 1 - 1e-6 must instead trip
    radius_overflow_risk under the "ball_norm_saturation" rule, with n_norm_saturated >= 1."""
    manifest, held = _write_small_tree(tmp_path)
    coords = _SMALL_TREE_BALL_COORDS.clone().double()
    coords[0] = torch.tensor([1.0 - 1e-7, 0.0], dtype=torch.float64)  # norm ~0.9999999, saturated
    ck = tmp_path / "ballsat_epoch1.pth"
    torch.save({"embeddings": coords}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"ballsat={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    row = json.loads(out.read_text())["arms"]["ballsat"]["checkpoints"][0]
    assert row["poincare_radius_source"] == "ball_coordinates"
    assert row["radius_guard"] == "ball_norm_saturation"
    assert row["radius_overflow_risk"] is True
    assert row["n_norm_saturated"] >= 1
    assert row["max_radius"] < 100.0  # the OLD threshold, confirmed structurally unreachable here
    assert "saturat" in rc.stdout.lower()


def test_radius_guard_silent_on_ball_coordinates_path_when_norms_are_comfortable(tmp_path):
    """The mirror case: every ball-coordinate norm comfortably inside the ball must NOT trip the
    saturation guard, and n_norm_saturated must be exactly 0."""
    manifest, held = _write_small_tree(tmp_path)
    ck = tmp_path / "ballsafe_epoch1.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"ballsafe={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    row = json.loads(out.read_text())["arms"]["ballsafe"]["checkpoints"][0]
    assert row["radius_guard"] == "ball_norm_saturation"
    assert row["radius_overflow_risk"] is False
    assert row["n_norm_saturated"] == 0


# --- Important 2 (fix round 1): vendrov_recall coverage that could actually fail ---


def test_vendrov_recall_core_finds_a_genuinely_visible_pair():
    """`vendrov_recall` itself is tautologically 0.0 for ANY input on the real P2 split -- the
    held_out array both removes edges from the visible graph and supplies the query, so every
    query's own edge is guaranteed excluded by construction. No fixture through that function can
    prove the encode+isin mechanism actually works. `_vendrov_recall_core` decouples "removed"
    from "queried": here only node 4's edge is removed from the visible graph, so node 3's edge
    (1, 3) is still visible and its query must correctly hit, giving a genuine, exactly-assertable
    non-zero recall of 0.5 -- an implementation that unconditionally returns 0.0 fails this."""
    # 0 -> {1, 2}; 1 -> {3, 4}
    parent = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    n_nodes = 5
    visible_anc, visible_desc = _visible_basic_edges(parent, held_out=np.array([4]), n_nodes=n_nodes)
    queries = np.array([3, 4], dtype=np.int64)
    true_parents = parent[queries]
    recall = _vendrov_recall_core(visible_anc, visible_desc, queries, true_parents, n_nodes)
    assert recall == 0.5

    # cross-checked against the independent reference implementation on the SAME fixture, so
    # this isn't just two copies of the same arithmetic agreeing with itself.
    mat = baselines.vendrov_closure_rule(visible_anc, visible_desc, queries=queries,
                                         candidates=true_parents)
    assert float(np.diag(mat).mean()) == 0.5


def test_vendrov_recall_agrees_with_the_reference_closure_rule():
    """The scorer's vectorized `vendrov_recall` and the reference `baselines.vendrov_closure_rule`
    (called with the full visible edge list, diagonal extracted) must agree exactly on the same
    fixture -- this is what stops the two implementations of one rule from drifting apart. (Both
    are expected to be 0.0 here, by construction on the real split -- see the module docstring --
    but this test is not a hardcoded-0.0 check: it fails if the scorer's manual pair-encoding
    diverges from the reference's independent set-membership check, e.g. a wrong `key` causing an
    encoding collision.)

    NOTE: on the real P2 split, `vendrov_recall` is tautologically 0.0 for ANY input -- the
    held_out array both removes edges from the visible graph and supplies the query, so every
    query's own edge is guaranteed excluded by construction (see the module docstring). This test
    only proves the two implementations AGREE with each other, both at 0.0; it is NOT the test
    that proves the underlying encode+isin mechanism actually works on a genuinely non-zero case --
    that is `test_vendrov_recall_core_finds_a_genuinely_visible_pair` above, which decouples
    "removed" from "queried" to get an exactly-assertable non-zero recall of 0.5. Do not read this
    test as a differential proof by itself; read it alongside that one."""
    # matches the tree fixture used by the end-to-end CLI test above
    parent = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    n_nodes = 7
    held_out = np.array([3, 5], dtype=np.int64)

    scorer_value = vendrov_recall(parent, held_out, n_nodes)

    visible_anc, visible_desc = _visible_basic_edges(parent, held_out, n_nodes)
    mat = baselines.vendrov_closure_rule(visible_anc, visible_desc, queries=held_out,
                                         candidates=parent[held_out])
    reference_value = float(np.diag(mat).mean())

    assert scorer_value == reference_value == 0.0


# --- Important 3 (fix round 1): 'val' must be counted, reported, and warned about ---


def test_n_val_is_recorded_and_warned_when_nonzero(tmp_path):
    manifest, held = _write_small_tree(tmp_path, test=(3,), val=(5,))
    ck = tmp_path / "ck_epoch1.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"arm={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    assert "1 'val' node" in rc.stdout  # a LOUD warning, not a silent drop
    res = json.loads(out.read_text())
    assert res["n_val"] == 1
    assert res["n_held"] == 1


def test_n_val_is_zero_and_silent_when_val_is_empty(tmp_path):
    manifest, held = _write_small_tree(tmp_path, test=(3, 5), val=())
    ck = tmp_path / "ck_epoch1.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"arm={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    assert "'val' node" not in rc.stdout
    res = json.loads(out.read_text())
    assert res["n_val"] == 0


# --- Minor 5 (fix round 1): NaN is not valid JSON -- emit null instead ---


def test_empty_heldout_metrics_are_null_not_nan_in_strict_json(tmp_path):
    """An empty held-out set makes every linkpred metric float('nan'); json.dumps's default
    allow_nan=True would emit the bare NaN token, which a strict (RFC 8259) parser rejects."""
    manifest, held = _write_small_tree(tmp_path, test=(), val=())
    ck = tmp_path / "ck_epoch1.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"arm={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    text = out.read_text()
    assert "NaN" not in text  # the bare token a strict JSON parser rejects

    def _reject_nonfinite(name):
        raise ValueError(f"strict JSON parser encountered non-finite constant: {name}")

    res = json.loads(text, parse_constant=_reject_nonfinite)  # raises if anything slipped through
    assert res["baselines"]["sibling_chance_mean"] is None
    assert res["arms"]["arm"]["checkpoints"][0]["metrics"]["mrr"] is None


# --- C1/C2 (review finding, 2026-09-24, p2_amendment_3_20260924): the scorer must emit the
# degree_prior and chance_mrr_mean/chance_hits_at_1_mean baselines alongside sibling_chance_mean.


def test_baselines_include_degree_prior_and_chance_mrr_mean(tmp_path):
    manifest, held = _write_small_tree(tmp_path)   # held=(3, 5), both pool size 2
    ck = tmp_path / "ck_epoch1.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS}, ck)

    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"arm={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    baselines = json.loads(out.read_text())["baselines"]
    assert "degree_prior" in baselines
    assert set(("mrr", "hits_at_1", "normalized_rank")) <= set(baselines["degree_prior"])
    assert "chance_mrr_mean" in baselines
    assert "chance_hits_at_1_mean" in baselines
    # both held nodes have pool size 2 (chance = H_2/2 = 0.75), which is ABOVE the hits@1 chance
    # rate (1/2 = 0.5) -- the C2 gap, reproduced on real scorer output, not just a unit fixture.
    assert baselines["chance_mrr_mean"] == pytest.approx(0.75)
    assert baselines["chance_hits_at_1_mean"] == pytest.approx(0.5)
    assert baselines["chance_mrr_mean"] > baselines["chance_hits_at_1_mean"]


# --- C5 (review finding, 2026-09-24): --closure overrides manifest['source_npz'], which
# build_p2_split.py records as an absolute macOS build-time path absent inside the LRZ container.


def test_closure_override_replaces_a_dead_manifest_source_npz(tmp_path):
    """LESION CHECK. manifest['source_npz'] points at a path that does NOT exist (mirroring the
    real defect: an absolute macOS path, absent inside the container). Without --closure this
    must fail; WITH --closure pointing at a real closure file elsewhere, scoring must succeed and
    record which path it actually used."""
    parent = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    depth = np.array([0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    pairs = closure_from_parent(parent, depth)
    real_closure = tmp_path / "real_closure.npz"
    pairs.save(real_closure)

    held = tmp_path / "heldout.npz"
    np.savez(held, test=np.array([3, 5], dtype=np.int64), val=np.array([], dtype=np.int64))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "clade": "synthetic", "band": [0, 99],
        "source_npz": str(tmp_path / "DOES_NOT_EXIST_macos_build_time_path.npz"),
    }))
    ck = tmp_path / "ck_epoch1.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS}, ck)
    out = tmp_path / "scores.json"

    rc_missing = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"arm={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc_missing.returncode != 0, "must fail without --closure: source_npz does not exist"

    rc_ok = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--closure", str(real_closure), "--checkpoints", f"arm={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc_ok.returncode == 0, rc_ok.stderr
    res = json.loads(out.read_text())
    assert res["closure_path_used"] == str(real_closure.resolve())
    assert res["arms"]["arm"]["checkpoints"][0]["metrics"]["mrr"] == 1.0


def test_missing_closure_argument_fails_fast_with_a_clear_message(tmp_path):
    """HEALTHY / control direction: --closure pointing at a file that does not exist must fail
    IMMEDIATELY with a clear message -- proving a pre-flight check catches it, not a deep
    traceback three frames into TrainingPairs.load."""
    manifest, held = _write_small_tree(tmp_path)
    ck = tmp_path / "ck_epoch1.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS}, ck)
    out = tmp_path / "scores.json"
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--closure", str(tmp_path / "nope.npz"), "--checkpoints", f"arm={ck}", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert rc.returncode != 0
    assert "does not exist" in rc.stderr
    assert "Traceback" not in rc.stderr


# --- C4 (review finding, 2026-09-24): scorer output -> merge_p2_scorer_outputs -> p2_verdict,
# end to end, against a REAL score_p2_linkpred.py subprocess (not a hand-authored fixture) --
# "a verifier you wrote shares your blind spot", so this replays the real scorer's own output
# shape through the parser exactly as helpers/check_p2_prereg_parses_real_scorer_output.py does.


def _write_roll_window(tmp_path, stem="roll"):
    """Write a FULL rolling window of `P2_ROLL_WINDOW` checkpoints and return the glob for it.

    C-C (2026-09-26): `p2_run_value` now applies `P2_ROLL_WINDOW` and RAISES on a roll set that
    is not exactly that long. These end-to-end tests previously registered ONE checkpoint under
    the `_roll` key, which no production run ever produces -- the trainer's rolling queue keeps
    the last 5 and `p2_lrz_score.sh` globs `${tag}_epoch*.pth`. A fixture that is unrepresentative
    in precisely the dimension under test is how the window went unapplied for as long as it did,
    so the fixture is corrected rather than the invariant relaxed.
    """
    for epoch in range(201 - P2_ROLL_WINDOW, 201):
        torch.save({"embeddings": _SMALL_TREE_BALL_COORDS, "epoch": epoch},
                   tmp_path / f"{stem}_epoch{epoch}.pth")
    return str(tmp_path / f"{stem}_epoch*.pth")


def test_scorer_output_flows_through_merge_and_the_verdict_engine_end_to_end(tmp_path):
    manifest, held = _write_small_tree(tmp_path, test=(3, 5))
    ck = tmp_path / "ck_epoch200.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS, "epoch": 200}, ck)
    roll = _write_roll_window(tmp_path)

    real_out = tmp_path / "real.json"
    rc_real = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"vis00_s0_ms={ck}", "--checkpoints", f"vis00_s0_roll={roll}",
         "--checkpoints", f"vis50_s0_ms={ck}", "--checkpoints", f"vis50_s0_roll={roll}",
         "--out", str(real_out)],
        capture_output=True, text=True,
    )
    assert rc_real.returncode == 0, rc_real.stderr

    control_out = tmp_path / "control.json"
    rc_control = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
         "--checkpoints", f"randomdag_s0_ms={ck}", "--checkpoints", f"randomdag_s0_roll={roll}",
         "--baselines-key", "baselines_randomdag", "--out", str(control_out)],
        capture_output=True, text=True,
    )
    assert rc_control.returncode == 0, rc_control.stderr

    real_result = json.loads(real_out.read_text())
    control_result = json.loads(control_out.read_text())
    assert "baselines" in real_result
    assert "baselines_randomdag" in control_result
    assert "baselines" not in control_result   # C4 #3: never the plain key for a control run

    merged = merge_p2_scorer_outputs([real_result, control_result])
    assert {"vis00_s0_ms", "vis00_s0_roll", "vis50_s0_ms", "vis50_s0_roll",
           "randomdag_s0_ms", "randomdag_s0_roll"} <= set(merged["arms"])

    v = p2_verdict(merged, seeds=(0,))
    # C-B (2026-09-26): `verdict in {the four possible verdicts}` cannot fail -- replaced with
    # the structural facts this test is actually about.
    assert v["amendment_1_applied"] is False        # the frozen single-shared-control reading
    assert "control" in v and "controls" not in v
    assert v["control"]["group"] == "randomdag"
    # the control's OWN measured floor (from control_result) is what the engine actually used --
    # never silently falling back to the real tree's baselines block (C4 #3's whole point).
    assert v["randomdag_sibling_chance_mean"] == pytest.approx(
        control_result["baselines_randomdag"]["sibling_chance_mean"])


# --- p2_amendment_4_20260924 + Part 2 (2026-09-24): the degree-matched control's name
# (degmatch_vis00/degmatch_vis50), amendment_2's matched-per-visibility shape, and Part 2's
# seed-tagged --baselines-key, all replayed against a REAL score_p2_linkpred.py subprocess
# across 2 seeds -- "a verifier you wrote shares your blind spot", so this exercises the actual
# CLI surface p2_lrz_score.sh drives, not just hand-authored fixtures.


def test_degmatch_and_seed_tagged_baselines_flow_through_merge_and_amendment_4_end_to_end(tmp_path):
    manifest, held = _write_small_tree(tmp_path, test=(3, 5))
    ck = tmp_path / "ck_epoch200.pth"
    torch.save({"embeddings": _SMALL_TREE_BALL_COORDS, "epoch": 200}, ck)
    roll = _write_roll_window(tmp_path)

    results = []
    for s in (0, 1):
        real_out = tmp_path / f"real_s{s}.json"
        rc_real = subprocess.run(
            [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
             "--checkpoints", f"vis00_s{s}_ms={ck}", "--checkpoints", f"vis00_s{s}_roll={roll}",
             "--checkpoints", f"vis50_s{s}_ms={ck}", "--checkpoints", f"vis50_s{s}_roll={roll}",
             "--baselines-key", f"baselines_s{s}", "--out", str(real_out)],
            capture_output=True, text=True,
        )
        assert rc_real.returncode == 0, rc_real.stderr
        results.append(json.loads(real_out.read_text()))

        for arm in ("degmatch_vis00", "degmatch_vis50"):
            control_out = tmp_path / f"{arm}_s{s}.json"
            rc_control = subprocess.run(
                [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", str(held),
                 "--checkpoints", f"{arm}_s{s}_ms={ck}", "--checkpoints", f"{arm}_s{s}_roll={roll}",
                 "--baselines-key", f"baselines_{arm}_s{s}", "--out", str(control_out)],
                capture_output=True, text=True,
            )
            assert rc_control.returncode == 0, rc_control.stderr
            results.append(json.loads(control_out.read_text()))

    for r in results:
        assert any(k.endswith(f"_s{i}") and k.startswith("baselines") for i in (0, 1) for k in r), (
            "every scorer output must carry a SEED-TAGGED baselines key, not the bare 'baselines'")

    # the seed-tagged keys never collide, so merging must not raise on agreement (both seeds
    # score the SAME tiny tree here, so the cross-seed spread is exactly 0 -- a trivial pass,
    # but proves the wiring end to end on real scorer output rather than a hand-built fixture).
    merged = merge_p2_scorer_outputs(results)
    assert "_baselines_cross_seed_spread" in merged
    assert merged["_baselines_cross_seed_spread"]     # non-empty: families were actually found

    v = p2_verdict(merged, seeds=(0, 1), amendment_1=True, amendment_2=True, amendment_4=True)
    assert v["amendment_4_applied"] is True
    # C-B (2026-09-26): the two assertions that used to stand here could not fail.
    # `v["verdict"] in {the four possible verdicts}` is true for every input and every
    # implementation of the reading logic, and `set(v["controls"]) <= {...}` is satisfied by the
    # EMPTY SET and by a single control -- blind to under-application, which is the `issubset`
    # lesion recurring one file over from
    # test_visibility_zero_keeps_parent_edges_plus_heldout_ancestry, where wave 3 had just fixed
    # it. Replaced with the verdict's own named booleans and an EQUALITY on the controls.
    assert set(v["controls"]) == {"degmatch_vis00", "degmatch_vis50"}
    # This fixture scores a TINY tree for a handful of epochs, so its runs do not clear the
    # validity gates and the reading is UNINFORMATIVE. Stating that is the point: the old
    # `verdict in {the four}` assertion passed while concealing it, so a change that made this
    # input suddenly readable -- or that broke the gates open -- would not have been noticed.
    # UNINFORMATIVE means no arm is read, hence no `by_arm_verdict`.
    assert v["verdict"] == "UNINFORMATIVE"
    assert "by_arm_verdict" not in v
    assert v["meaning"].startswith("A validity gate failed")
    # each control's published floor must be ITS OWN measured value, never the real tree's
    for control in ("degmatch_vis00", "degmatch_vis50"):
        own = merged[f"baselines_{control}_s0"]["sibling_chance_mean"]
        assert v["randomdag_sibling_chance_mean"][control] == pytest.approx(own)
        assert merged["_control_baselines_present"][control] == f"baselines_{control}_s0"
