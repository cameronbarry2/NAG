import torch
from diffusers.models.attention import CrossAttention
from functools import wraps

def _wrap_forward(orig_forward, guidance_scale: float, normalize: bool):
    @wraps(orig_forward)
    def new_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # Run original cross-attention to get output & attention weights
        # NOTE: Diffusers’ CrossAttention returns just hidden_states, so to actually
        # capture weights you’d need to dive into its internals. This stub
        # shows where you’d insert the NAG math.
        out = orig_forward(
            hidden_states, encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask, **cross_attention_kwargs
        )
        # PSEUDO‐CODE:
        # attn_scores = self.get_last_attention_maps()
        # if normalize:
        #     attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + 1e-5)
        # attn_scores = attn_scores * guidance_scale
        # self.set_attention_maps(attn_scores)
        return out
    return new_forward

def apply_nag_to_unet(unet: torch.nn.Module, guidance_scale: float = 1.5, normalize: bool = True):
    """
    Monkey-patch every CrossAttention.forward in the UNet to apply NAG.
    """
    for module in unet.modules():
        if isinstance(module, CrossAttention):
            orig = module.forward
            module.forward = _wrap_forward(orig, guidance_scale, normalize)