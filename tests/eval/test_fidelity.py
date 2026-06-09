import numpy as np
from taxembed.eval.fidelity import multiplicative_distortion, knn_retrieval_precision


def test_distortion_perfect_when_proportional():
    d_emb = np.array([1.0, 2.0, 4.0])
    d_tree = np.array([2.0, 4.0, 8.0])      # exactly 0.5x — proportional
    out = multiplicative_distortion(d_emb, d_tree)
    # multiplicative distortion is scale-invariant after global rescale -> ~1.0
    assert abs(out["median"] - 1.0) < 1e-9


def test_distortion_detects_disorder():
    d_emb = np.array([1.0, 5.0, 2.0])
    d_tree = np.array([1.0, 2.0, 5.0])      # ranks scrambled
    out = multiplicative_distortion(d_emb, d_tree)
    assert out["median"] > 1.0


def test_knn_retrieval_precision_perfect_and_chance():
    # 4 query nodes; emb-NN order identical to tree-NN order -> precision 1.0
    # distance matrices (row=query, col=candidate), diagonal = self = inf
    INF = np.inf
    d_tree = np.array([[INF, 1, 2, 3],
                       [1, INF, 2, 3],
                       [2, 1, INF, 3],
                       [3, 2, 1, INF]], float)
    d_emb = d_tree.copy()
    p = knn_retrieval_precision(d_emb, d_tree, k=1)
    assert np.allclose(p, 1.0)
    # if emb distances are reversed, precision at k=1 should drop
    p_bad = knn_retrieval_precision(d_tree[:, ::-1].copy(), d_tree, k=1)
    assert p_bad.mean() < 1.0
