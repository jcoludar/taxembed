"""Unit tests for the negative-hardness diagnostic helpers (Phase 1, E1c)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import pytest

from _negative_hardness import numpy_poincare_distance, softmax_pj, label_negatives


def test_poincare_distance_origin():
    # Distance from origin to itself is 0; symmetric; grows toward the boundary.
    o = np.zeros(3)
    assert numpy_poincare_distance(o, o) == pytest.approx(0.0, abs=1e-6)
    a = np.array([0.5, 0.0, 0.0])
    b = np.array([-0.5, 0.0, 0.0])
    d_ab = numpy_poincare_distance(a, b)
    assert d_ab > 0
    assert numpy_poincare_distance(b, a) == pytest.approx(d_ab, rel=1e-6)
    # Closer-to-boundary pair is farther than a near-origin pair of equal Euclidean gap.
    near = numpy_poincare_distance(np.array([0.0, 0, 0]), np.array([0.1, 0, 0]))
    far = numpy_poincare_distance(np.array([0.85, 0, 0]), np.array([0.95, 0, 0]))
    assert far > near


def test_softmax_pj_mass():
    # Positive much closer than negatives -> negatives get ~0 p_j mass.
    p_neg = softmax_pj(np.array([0.1]), np.array([[5.0, 5.0, 5.0]]))
    assert p_neg.shape == (1, 3)
    assert p_neg.sum() < 0.05
    # Positive far, negatives near -> negatives carry almost all the mass.
    p_neg2 = softmax_pj(np.array([5.0]), np.array([[0.1, 0.1, 0.1]]))
    assert p_neg2.sum() > 0.9
    # Within a row, the nearer negative gets more mass than the farther one.
    p_neg3 = softmax_pj(np.array([3.0]), np.array([[0.5, 4.0]]))
    assert p_neg3[0, 0] > p_neg3[0, 1]


def test_label_negatives():
    # node_class_arr / node_gp_arr indexed by node id; -1 = "no class/gp".
    node_class = np.array([-1, 10, 10, 20, 20, -1], dtype=np.int64)
    node_gp = np.array([-1, 100, 100, 200, 999, -1], dtype=np.int64)
    descendants = np.array([1, 3], dtype=np.int64)        # batch of 2 anchors
    negatives = np.array([[2, 4, 5],                       # for desc 1 (class 10, gp 100)
                          [4, 1, 0]], dtype=np.int64)      # for desc 3 (class 20, gp 200)
    within_class, within_gp = label_negatives(node_class, node_gp, descendants, negatives)
    # desc 1: neg 2 shares class 10 AND gp 100; neg 4 class 20 (no); neg 5 class -1 (no)
    assert within_class[0].tolist() == [True, False, False]
    assert within_gp[0].tolist() == [True, False, False]
    # desc 3 (class 20, gp 200): neg 4 shares class 20 but gp 999 (class yes, gp no);
    #                            neg 1 class 10 (no); neg 0 class -1 (no)
    assert within_class[1].tolist() == [True, False, False]
    assert within_gp[1].tolist() == [False, False, False]
    # -1 anchors must never count as within-clade (guard against sentinel collisions)
    desc_noclass = np.array([0], dtype=np.int64)
    negs = np.array([[5, 5, 5]], dtype=np.int64)           # also class -1
    wc, wg = label_negatives(node_class, node_gp, desc_noclass, negs)
    assert wc.sum() == 0 and wg.sum() == 0


def test_poincare_distance_closedform_and_broadcast():
    # closed form: d(0, x) = 2*arctanh(|x|)
    x = np.array([0.3, 0.0, 0.0])
    assert numpy_poincare_distance(np.zeros(3), x) == pytest.approx(2 * np.arctanh(0.3), rel=1e-6)
    # broadcast (B,1,dim) vs (B,n,dim) -> (B,n), the driver's actual path
    anc = np.array([[[0.1, 0, 0]], [[0.2, 0, 0]]])              # (2,1,3)
    negs = np.array([[[0.3, 0, 0], [0.4, 0, 0]],
                     [[0.5, 0, 0], [0.6, 0, 0]]])                # (2,2,3)
    d = numpy_poincare_distance(anc, negs)
    assert d.shape == (2, 2)
    assert d[0, 0] < d[0, 1] and d[1, 0] < d[1, 1]


def test_softmax_pj_multirow_and_normalization():
    d_pos = np.array([0.1, 5.0, 2.0])
    d_neg = np.array([[5.0, 5.0, 5.0], [0.1, 5.0, 0.1], [1.0, 2.0, 3.0]])
    p = softmax_pj(d_pos, d_neg)
    assert p.shape == (3, 3)
    pos_share = 1.0 - p.sum(axis=1)                              # implied positive column
    assert np.all(pos_share > -1e-12) and np.all(pos_share < 1.0 + 1e-12)
    assert p[0].sum() < 0.05 and p[1].sum() > 0.9                # row independence
