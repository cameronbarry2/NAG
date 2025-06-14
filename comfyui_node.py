import torch
from comfy import node
from .nag import apply_nag_embeds

@node(
    name="Dual NAG",
    category="Flux / Guidance",
    version="1.0"
)
class DualNAGNode:
    """
    Applies Normalized Attention Guidance separately to positive and negative embeddings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Positive branch
                "pos_original_text_embeds": ("TENSOR",),
                "pos_nag_text_embeds": ("TENSOR",),
                "pos_nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "pos_nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "pos_nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Negative branch
                "neg_original_text_embeds": ("TENSOR",),
                "neg_nag_text_embeds": ("TENSOR",),
                "neg_nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "neg_nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "neg_nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR")
    FUNCTION = "apply_dual_nag"
    CATEGORY = "Flux / Guidance"

    def apply_dual_nag(
        self,
        pos_original_text_embeds: torch.Tensor,
        pos_nag_text_embeds: torch.Tensor,
        pos_nag_scale: float,
        pos_nag_tau: float,
        pos_nag_alpha: float,
        neg_original_text_embeds: torch.Tensor,
        neg_nag_text_embeds: torch.Tensor,
        neg_nag_scale: float,
        neg_nag_tau: float,
        neg_nag_alpha: float,
    ):
        pos_out = apply_nag_embeds(
            pos_original_text_embeds,
            pos_nag_text_embeds,
            pos_nag_scale,
            pos_nag_tau,
            pos_nag_alpha,
        )
        neg_out = apply_nag_embeds(
            neg_original_text_embeds,
            neg_nag_text_embeds,
            neg_nag_scale,
            neg_nag_tau,
            neg_nag_alpha,
        )
        return (pos_out, neg_out)