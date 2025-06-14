import torch

def apply_nag_embeds(
    orig_embeds: torch.Tensor,
    nag_embeds: torch.Tensor,
    scale: float,
    tau: float,
    alpha: float,
) -> torch.Tensor:
    """
    Simple Normalized Attention Guidance on embedding vectors.
    - Compute diff = nag_embeds - orig_embeds
    - Normalize diff per vector
    - Raise magnitude to tau
    - Scale and blend with alpha
    """
    # difference
    diff = nag_embeds - orig_embeds
    # normalize per-token diff
    norm_diff = diff / (diff.norm(dim=-1, keepdim=True) + 1e-5)
    # apply tau exponent
    diff_tau = torch.sign(norm_diff) * torch.abs(norm_diff) ** tau
    # scale
    scaled = diff_tau * scale
    # blend back with alpha
    return orig_embeds + scaled * alpha