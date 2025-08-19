import torch
import numpy as np

# ------------------------------------------------------
# Transform between ALR and Simplex space with bounds
# ------------------------------------------------------
def compute_alr_bounds_from_simplex_bounds(
    lower: torch.Tensor,   # (4,)
    upper: torch.Tensor,   # (4,)
    num_samples: int = 20000,
    batch: int = 5000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection‐sample inside [lower,upper] ∩ simplex, then ALR→min/max."""
    import numpy as _np

    accepted = []
    lower_np, upper_np = lower.cpu().numpy(), upper.cpu().numpy()
    while sum(a.shape[0] for a in accepted) < num_samples:
        Xs = _np.random.dirichlet(_np.ones(4), size=batch)
        mask = (Xs >= lower_np).all(axis=1) & (Xs <= upper_np).all(axis=1)
        if mask.any():
            accepted.append(Xs[mask])
    Xall = _np.vstack(accepted)[:num_samples]
    Z = np.log(Xall[:, :3] / Xall[:, [3]])  # ALR on the 3 log-ratios
    alr_lb = torch.from_numpy(Z.min(axis=0)).to(lower)
    alr_ub = torch.from_numpy(Z.max(axis=0)).to(lower)
    return alr_lb, alr_ub

def alr_to_simplex(z: torch.Tensor) -> torch.Tensor:
    """Invert ALR: z∈ℝ³ → x∈Δ³ (4‐dim simplex)."""
    exp_z = torch.exp(z)
    ones  = torch.ones(z.size(0), 1, dtype=z.dtype, device=z.device)
    num   = torch.cat([exp_z, ones], dim=1)
    return num / num.sum(dim=1, keepdim=True)