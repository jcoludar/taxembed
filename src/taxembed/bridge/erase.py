"""CLEAN side (spec §6): LEACE erasure tied to f's continuous tangent targets, a variance-AND-concept
matched specificity control, and the principal-angle overlap that MEASURES READ<->CLEAN coherence."""
from __future__ import annotations
import numpy as np
from concept_erasure import LeaceEraser


def leace_erase(X: np.ndarray, Z: np.ndarray, dtype=np.float64):
    """Erase the subspace linearly predicting concept Z (continuous targets OK). Returns (X_erased, P)
    where P is the (d,d) linear part s.t. X_erased ≈ (X-μ)@P + μ. P is recovered by probing the AFFINE
    eraser: for any affine er(x)=(x-c)P+c, er(e_i+μ)-er(μ)=e_i·P — independent of the eraser's internal
    centre c. All calls go through one `_apply` so dtype is consistent. If the installed LeaceEraser
    rejects float64, pass dtype=np.float32 — the affineness self-check still guards correctness."""
    import torch
    X = np.asarray(X, dtype); Z = np.asarray(Z, dtype)
    if Z.ndim == 1:
        Z = Z[:, None]
    tdt = torch.float64 if dtype == np.float64 else torch.float32
    er = LeaceEraser.fit(torch.tensor(X), torch.tensor(Z))
    def _apply(M):
        return er(torch.as_tensor(np.asarray(M, dtype), dtype=tdt)).cpu().numpy().astype(np.float64)
    Xe = _apply(X)
    d = X.shape[1]; mu = X.mean(0)
    x0 = X[:1]
    resid = _apply(2 * x0) - 2 * _apply(x0) + _apply(np.zeros_like(x0))   # affineness self-check
    assert np.allclose(resid, 0.0, atol=1e-4), "LeaceEraser not affine on this version — inspect the API"
    P = _apply(np.eye(d) + mu) - _apply(np.tile(mu, (d, 1)))
    return Xe, P


def erasing_projector_rowspace(P: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Orthonormal basis of the ERASED directions = column space of (I - P)."""
    M = np.eye(P.shape[0]) - P
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > tol * max(s[0], 1e-12)]


def apply_projector(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    mu = X.mean(0)
    return (X - mu) @ P + mu


def matched_control_subspace(X, target, avoid, erased_energy, max_rank=None) -> np.ndarray:
    """Specificity control (spec §6.3): erase a subspace that is target(function)-correlated,
    avoid(taxonomy)-ORTHOGONAL, AND carries data-variance ≈ `erased_energy` (the variance LEACE
    removed) — matched on BOTH variance and concept."""
    X = np.asarray(X, np.float64); Xc = X - X.mean(0)
    from sklearn.linear_model import Ridge
    wt = np.atleast_2d(Ridge(alpha=1.0).fit(X, np.asarray(target, np.float64)).coef_).T
    wa = np.atleast_2d(Ridge(alpha=1.0).fit(X, np.asarray(avoid, np.float64)).coef_).T
    Qa = np.linalg.qr(wa)[0]
    cand = np.linalg.qr(wt - Qa @ (Qa.T @ wt))[0]   # function dirs ⟂ taxonomy, orthonormal candidates
    var = (Xc @ cand).var(0); order = np.argsort(-var)   # greedily accumulate to the target variance
    cum, picked = 0.0, []
    for j in order:
        if max_rank and len(picked) >= max_rank:
            break
        picked.append(int(j)); cum += var[j]
        if cum >= erased_energy:
            break
    U = cand[:, picked]
    return np.eye(X.shape[1]) - U @ U.T


def principal_angle_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean cos^2 of principal angles between subspaces spanned by columns of A,B (1=identical,0=orthog)."""
    Qa = np.linalg.qr(A)[0]; Qb = np.linalg.qr(B)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))
