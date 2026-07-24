"""The Taxonomy Bridge: align protein representations to a Poincaré taxonomy embedding,
READ (place) and expose the taxonomy subspace to CLEAN. Pure numpy/scipy; torch only to load
the checkpoint. Reuses taxembed.eval.treedist for exact tree distances.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from taxembed.eval.treedist import TreeDistance


def exp0(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Exponential map at the origin: exp0(v) = tanh(||v||) v/||v||."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.tanh(n) / np.maximum(n, eps)
    return v * scale


def log0(p: np.ndarray, eps: float = 1e-12, max_norm: float = 1 - 1e-7) -> np.ndarray:
    """Logarithmic map at the origin: log0(p) = arctanh(||p||) p/||p||."""
    p = np.asarray(p, dtype=np.float64)
    n = np.linalg.norm(p, axis=-1, keepdims=True)
    nc = np.minimum(n, max_norm)
    scale = np.arctanh(nc) / np.maximum(n, eps)
    return p * scale


class TaxonomyEmbedding:
    """Loads the LOCKED metazoa embedding: positions (N,d) + taxid<->idx + parent/depth tree."""

    def __init__(self, ckpt_path, mapping_path, edgelist_path, tensor_key: str = "embeddings"):
        # Accept both checkpoint formats: the released .safetensors (HF, tensor key "embedding") and
        # the training .pth (torch.save, key "embeddings"). Each falls back to the other key name.
        if str(ckpt_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(str(ckpt_path))
            tensor = sd["embedding"] if "embedding" in sd else sd["embeddings"]
        else:
            obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            tensor = obj[tensor_key] if tensor_key in obj else obj["embedding"]
        pos = np.asarray(tensor, dtype=np.float64)
        norms = np.linalg.norm(pos, axis=1)
        if (norms >= 1.0).any():                       # guard: must be ball positions, not tangents
            raise ValueError(f"{tensor_key}: {(norms>=1).sum()} norms >= 1; wrong tensor?")
        self.positions = pos                            # (N, d) in the open unit ball
        self.n_nodes, self.dim = pos.shape

        self.taxid2idx: dict[int, int] = {}
        self.idx2taxid = np.full(self.n_nodes, -1, dtype=np.int64)
        for ln in Path(mapping_path).read_text().splitlines()[1:]:
            tid, idx = ln.split("\t")
            self.taxid2idx[int(tid)] = int(idx)
            self.idx2taxid[int(idx)] = int(tid)

        parent = np.arange(self.n_nodes, dtype=np.int64)   # default self-parent (root)
        for ln in Path(edgelist_path).read_text().splitlines():
            p, c = ln.split()
            parent[int(c)] = int(p)
        self.parent = parent
        self.depth = self._compute_depth(parent)
        self.tree = TreeDistance(self.parent, self.depth)

    @staticmethod
    def _compute_depth(parent: np.ndarray) -> np.ndarray:
        n = len(parent)
        depth = np.full(n, -1, dtype=np.int64)
        children: dict[int, list[int]] = {}
        roots = []
        for c, p in enumerate(parent):
            if p == c:
                roots.append(c)
            else:
                children.setdefault(int(p), []).append(c)
        stack = [(r, 0) for r in roots]
        while stack:
            node, d = stack.pop()
            depth[node] = d
            for ch in children.get(node, ()):
                stack.append((ch, d + 1))
        return depth

    def idx_of_taxid(self, taxid: int) -> int | None:
        return self.taxid2idx.get(int(taxid))

    def position_of_taxid(self, taxid: int) -> np.ndarray:
        return self.positions[self.taxid2idx[int(taxid)]]


def poincare_distance_matrix(q: np.ndarray, r: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, np.float64); r = np.asarray(r, np.float64)
    qq = (q * q).sum(1)[:, None]; rr = (r * r).sum(1)[None, :]
    sq = qq + rr - 2.0 * (q @ r.T)     # ||q-r||^2 = qq + rr - 2 q·r  (broadcast (Q,1)+(1,R))
    denom = np.maximum((1 - qq) * (1 - rr), eps)
    return np.arccosh(np.maximum(1.0 + 2.0 * sq / denom, 1.0))


def retrieve_nearest(q: np.ndarray, r: np.ndarray, k: int = 1, chunk: int = 4096) -> np.ndarray:
    """Return (Q,k) indices into r of the k nearest by hyperbolic distance (chunked over r)."""
    q = np.asarray(q, np.float64)
    best_d = np.full((len(q), k), np.inf); best_i = np.zeros((len(q), k), dtype=np.int64)
    for s in range(0, len(r), chunk):
        block = r[s:s + chunk]
        D = poincare_distance_matrix(q, block)                      # (Q, b)
        cat_d = np.concatenate([best_d, D], axis=1)
        cat_i = np.concatenate([best_i, np.broadcast_to(np.arange(s, s + len(block)), (len(q), len(block)))], axis=1)
        order = np.argsort(cat_d, axis=1)[:, :k]
        best_d = np.take_along_axis(cat_d, order, axis=1)
        best_i = np.take_along_axis(cat_i, order, axis=1)
    return best_i


def pca_reduce(X: np.ndarray, dim: int):
    p = PCA(n_components=min(dim, X.shape[1], X.shape[0]), svd_solver="full").fit(X)
    return p

class Bridge:
    """Linear bridge: PCA(rep) -> Ridge -> tangent at origin -> exp0 -> Poincaré position."""
    def __init__(self, pca, ridge, train_tangent_mean):
        self.pca, self.ridge, self._mu = pca, ridge, train_tangent_mean

    @classmethod
    def align(cls, reps, positions, pca_dim: int = 64, alpha: float = 1.0):
        reps = np.asarray(reps, np.float64); positions = np.asarray(positions, np.float64)
        pca = pca_reduce(reps, pca_dim)
        Z = pca.transform(reps)
        Y = log0(positions)                                  # regress in tangent space
        ridge = Ridge(alpha=alpha, fit_intercept=True).fit(Z, Y)
        return cls(pca, ridge, Y.mean(0))

    def place(self, reps) -> np.ndarray:
        Z = self.pca.transform(np.asarray(reps, np.float64))
        return exp0(self.ridge.predict(Z))                   # back to the ball

    def tax_subspace(self, energy: float = 0.95, rank: int | None = None) -> np.ndarray:
        """Orthonormal basis (PCA-dim, k) of the LOW-RANK predictive subspace — the directions in
        PCA-rep space that carry the taxonomy signal (what CLEAN erases). Truncated by cumulative
        singular-value energy (default 95%) or an explicit `rank`.
        LOAD-BEARING (adversarial-review B1): ridge.coef_ is full row-rank (= pca_dim), so an
        un-truncated basis spans the WHOLE pca space -> principal_angle_overlap vs any subspace == 1.0
        and the coherence claim becomes unfalsifiable. The truncation is the fix, not cosmetic."""
        W = self.ridge.coef_.T                               # (pca_dim, tax_dim)
        U, s, _ = np.linalg.svd(W, full_matrices=False)
        if rank is None:
            cum = np.cumsum(s ** 2) / np.sum(s ** 2)
            rank = int(np.searchsorted(cum, energy) + 1)
        return U[:, :rank]
