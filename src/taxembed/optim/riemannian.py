"""
Riemannian Adam optimizer for Poincaré ball embeddings.

Euclidean gradients must be scaled by the inverse of the Poincaré metric tensor
to become valid Riemannian gradients. The Poincaré ball has conformal factor:

    g_x = (2 / (1 - ||x||^2))^2 * I

So the Riemannian gradient is:

    grad_R = ((1 - ||x||^2)^2 / 4) * grad_E

See: hype/manifolds/poincare.py:41-53 (Facebook's original implementation)
"""

import torch


class RiemannianAdam(torch.optim.Adam):
    """Adam optimizer with Riemannian gradient correction for the Poincaré ball.

    Before each Adam step, Euclidean gradients are rescaled by the Poincaré
    conformal factor ((1 - ||p||^2)^2 / 4). After each step, parameters are
    projected back into the ball via retraction to max_norm.
    """

    def __init__(self, params, lr=1e-3, max_norm=0.999, **kwargs):
        self.max_norm = max_norm
        super().__init__(params, lr=lr, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        # Scale Euclidean gradients -> Riemannian gradients
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
                # Clamp to avoid division issues at the boundary
                p_sqnorm = torch.clamp(p_sqnorm, 0, 1 - 1e-5)
                conformal = (1 - p_sqnorm) ** 2 / 4
                p.grad.data.mul_(conformal.expand_as(p.grad.data))

        # Standard Adam step on the rescaled gradients
        loss = super().step(closure)

        # Retraction: project back into the Poincaré ball
        for group in self.param_groups:
            for p in group["params"]:
                norms = p.data.norm(dim=-1, keepdim=True)
                mask = norms >= self.max_norm
                scale = torch.where(
                    mask,
                    self.max_norm / (norms + 1e-8),
                    torch.ones_like(norms),
                )
                p.data.mul_(scale)

        return loss
