# Normalized Attention Guidance (Dual)

This extension adds a **Dual NAG** node to ComfyUI, letting you control NAG for both positive and negative text embeddings in one spot.

## Installation

1. Copy the entire `NormalizedAttentionGuidance/` folder into your ComfyUI `custom_nodes/` directory.
2. Restart ComfyUI.

## Usage

1. In your graph, add the **Dual NAG** node (category **Flux / Guidance**).
2. Feed in:
   - **pos_original_text_embeds** and **pos_nag_text_embeds**
   - **neg_original_text_embeds** and **neg_nag_text_embeds**
3. Tweak the six NAG parameters:
   - **pos_nag_scale**, **pos_nag_tau**, **pos_nag_alpha**
   - **neg_nag_scale**, **neg_nag_tau**, **neg_nag_alpha**
4. Use the two outputs **pos_text_embeds** and **neg_text_embeds** in your sampler.

This node applies the NAG algorithm to each embedding pair separately, blending original ↔ NAG embeddings with your chosen scale, τ and α.