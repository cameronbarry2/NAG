# Normalized Attention Guidance for ComfyUI

This extension adds a “Normalized Attention Guidance” node to ComfyUI, so you can inject the NAG algorithm into your UNet before sampling.

## Installation

1. Copy the entire `NormalizedAttentionGuidance/` folder into your ComfyUI `custom_nodes/` directory.
2. Restart ComfyUI.

## Usage

1. In your graph, add the **Normalized Attention Guidance** node (category “Flux / Guidance”).
2. Plug your UNet output into **Unet**.
3. Adjust **Guidance Scale** and **Normalize**.
4. Plug the resulting **Unet** into your sampler (KSampler, DDIM, etc.).

The node will monkey-patch every `CrossAttention` in the UNet to apply normalized attention scaling at inference time.
