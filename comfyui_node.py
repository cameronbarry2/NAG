import torch
from comfy.model_management import register_unet_modifier
from comfy import node
from .nag import apply_nag_to_unet

@node(
    name="Normalized Attention Guidance",
    category="Flux / Guidance",
    version="1.0"
)
class NormalizedAttentionGuidanceNode:
    """
    A Flux node that takes a UNet, applies NAG, and returns the modified UNet.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet": ("UNET",),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("UNET",)
    FUNCTION = "apply"
    CATEGORY = "Flux / Guidance"

    def apply(self, unet, guidance_scale, normalize):
        # apply the NAG monkey-patch to the model
        apply_nag_to_unet(unet, guidance_scale, normalize)
        return (unet,)