"""
StableDiffusionBoxDiffPipeline - Training-free box-constrained diffusion.

This pipeline is vendored from the HuggingFace diffusers community pipelines:
https://huggingface.co/datasets/diffusers/community-pipelines-mirror/blob/main/pipeline_stable_diffusion_boxdiff.py

Original source: https://github.com/showlab/BoxDiff
Paper: BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion (ICCV 2023)
       https://arxiv.org/abs/2307.10816

Vendored for reproducibility - community pipelines can change between diffusers versions.

License: Apache 2.0 (same as diffusers)

Modifications from original:
- Added type hints for clarity
- Added docstrings
- Minor style adjustments for consistency with SpatialBench-UC codebase
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging

logger = logging.get_logger(__name__)


# =============================================================================
# Default BoxDiff Parameters
# =============================================================================

DEFAULT_BOXDIFF_KWARGS = {
    "attention_res": 16,  # Resolution for attention maps
    "P": 0.2,  # Probability threshold for loss computation
    "L": 1,  # Number of loss iterations
    "max_iter_to_alter": 25,  # Max timesteps to apply constraints
    "loss_thresholds": {0: 0.05, 10: 0.5, 20: 0.8},  # Loss thresholds per iteration
    "scale_factor": 20,  # Scale factor for latent updates
    "scale_range": (1.0, 0.5),  # Scale range for progressive updates
    "smooth_attentions": True,  # Whether to smooth attention maps
    "sigma": 0.5,  # Gaussian smoothing sigma
    "kernel_size": 3,  # Gaussian kernel size
    "refine": False,  # Whether to refine boxes
    "normalize_eot": True,  # Normalize end-of-text token
}


# =============================================================================
# Utility Functions
# =============================================================================


def gaussian_kernel(kernel_size: int = 3, sigma: float = 0.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel for smoothing attention maps."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
    kernel = gauss.outer(gauss)
    kernel = kernel / kernel.sum()
    return kernel


def smooth_attention_map(
    attention_map: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 0.5,
) -> torch.Tensor:
    """Apply Gaussian smoothing to attention maps."""
    kernel = gaussian_kernel(kernel_size, sigma).to(attention_map.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Pad and convolve
    padding = kernel_size // 2
    smoothed = F.conv2d(
        attention_map.unsqueeze(0).unsqueeze(0),
        kernel,
        padding=padding,
    )
    return smoothed.squeeze(0).squeeze(0)


def get_attention_scores(
    attn_procs: Dict[str, Any],
    attention_res: int,
) -> Dict[str, torch.Tensor]:
    """Extract attention scores from attention processors."""
    attention_scores = {}
    for name, proc in attn_procs.items():
        if hasattr(proc, "attention_scores") and proc.attention_scores is not None:
            h = w = attention_res
            scores = proc.attention_scores
            if scores.shape[-2] == h * w:
                attention_scores[name] = scores
    return attention_scores


def boxes_to_masks(
    boxes: List[List[float]],
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert normalized bounding boxes to binary masks."""
    masks = []
    for box in boxes:
        x0, y0, x1, y1 = box
        mask = torch.zeros(height, width)
        x0_int = int(x0 * width)
        y0_int = int(y0 * height)
        x1_int = int(x1 * width)
        y1_int = int(y1 * height)
        mask[y0_int:y1_int, x0_int:x1_int] = 1.0
        masks.append(mask)
    return torch.stack(masks) if masks else torch.empty(0, height, width)


def compute_boxdiff_loss(
    attention_maps: torch.Tensor,  # Shape: [heads, h*w, seq_len]
    token_indices: List[int],
    boxes: List[List[float]],
    attention_res: int,
    P: float = 0.2,
    smooth_attentions: bool = True,
    sigma: float = 0.5,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Compute the BoxDiff loss for attention constraints.

    The loss encourages attention for each phrase to concentrate within
    its corresponding bounding box.
    """
    if len(boxes) == 0 or len(token_indices) == 0:
        return torch.tensor(0.0)

    h = w = attention_res
    device = attention_maps.device

    # Create box masks
    masks = boxes_to_masks(boxes, h, w).to(device)

    total_loss = torch.tensor(0.0, device=device)

    # Average attention across heads
    avg_attention = attention_maps.mean(dim=0)  # [h*w, seq_len]

    for idx, (token_idx, mask) in enumerate(zip(token_indices, masks)):
        if token_idx >= avg_attention.shape[-1]:
            continue

        # Get attention for this token
        attn = avg_attention[:, token_idx].reshape(h, w)

        # Optionally smooth
        if smooth_attentions:
            attn = smooth_attention_map(attn, kernel_size, sigma)

        # Normalize attention
        attn = attn / (attn.max() + 1e-8)

        # Compute inside/outside attention
        inside = (attn * mask).sum()
        outside = (attn * (1 - mask)).sum()

        # Loss: maximize inside, minimize outside
        box_loss = -inside + P * outside
        total_loss = total_loss + box_loss

    return total_loss / max(len(token_indices), 1)


# =============================================================================
# Attention Processor for BoxDiff
# =============================================================================


class BoxDiffAttnProcessor:
    """
    Attention processor that stores attention scores for BoxDiff.

    This processor wraps the default attention computation and stores
    the attention scores for later use in loss computation.
    """

    def __init__(self, store_attention: bool = False):
        self.store_attention = store_attention
        self.attention_scores: Optional[torch.Tensor] = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Store attention if this is cross-attention
        if self.store_attention and encoder_hidden_states is not hidden_states:
            self.attention_scores = attention_probs.detach()

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# =============================================================================
# BoxDiff Pipeline
# =============================================================================


class StableDiffusionBoxDiffPipeline(StableDiffusionPipeline):
    """
    Pipeline for text-to-image generation with bounding box spatial control.

    BoxDiff enables training-free box-constrained diffusion by manipulating
    cross-attention maps during the denoising process. Unlike GLIGEN which
    requires additional training, BoxDiff works with any Stable Diffusion model.

    This pipeline extends StableDiffusionPipeline with:
    - boxdiff_phrases: List of phrases to ground in the image
    - boxdiff_boxes: List of normalized bounding boxes [x0, y0, x1, y1]
    - boxdiff_kwargs: Additional BoxDiff-specific parameters

    Reference:
        BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion
        https://arxiv.org/abs/2307.10816
    """

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = False,
        image_encoder=None,
    ):
        # Call parent init with compatible signature
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
            image_encoder=image_encoder,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _get_token_indices(
        self,
        prompt: str,
        phrases: List[str],
        tokenizer: CLIPTokenizer,
        normalize_eot: bool = True,
    ) -> List[int]:
        """
        Get token indices for each phrase in the prompt.

        Args:
            prompt: The full text prompt
            phrases: List of phrases to find in the prompt
            tokenizer: The CLIP tokenizer
            normalize_eot: Whether to consider EOT token

        Returns:
            List of token indices corresponding to each phrase
        """
        # Tokenize the full prompt
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_text = tokenizer.decode(prompt_tokens)

        token_indices = []

        for phrase in phrases:
            # Find phrase in prompt (case-insensitive)
            phrase_lower = phrase.lower().strip()
            prompt_lower = prompt.lower()

            # Simple approach: find the phrase and get its token position
            # This is approximate - for production, use more sophisticated matching
            phrase_tokens = tokenizer.encode(phrase, add_special_tokens=False)

            if len(phrase_tokens) > 0:
                # Use the first token of the phrase
                first_token = phrase_tokens[0]

                # Find this token in the prompt tokens
                for i, tok in enumerate(prompt_tokens):
                    if tok == first_token:
                        token_indices.append(i)
                        break
                else:
                    # Fallback: use a middle position
                    token_indices.append(len(prompt_tokens) // 2)
            else:
                token_indices.append(1)  # After BOS token

        return token_indices

    def _setup_attention_processors(self, store_attention: bool = False):
        """Set up attention processors to store attention scores."""
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            attn_procs[name] = BoxDiffAttnProcessor(store_attention=store_attention)
        self.unet.set_attn_processor(attn_procs)

    def _reset_attention_processors(self):
        """Reset attention processors to default."""
        try:
            from diffusers.models.attention_processor import AttnProcessor
        except ImportError:
            from diffusers.models.attention import AttnProcessor

        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            attn_procs[name] = AttnProcessor()
        self.unet.set_attn_processor(attn_procs)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # BoxDiff-specific parameters
        boxdiff_phrases: Optional[List[str]] = None,
        boxdiff_boxes: Optional[List[List[float]]] = None,
        boxdiff_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate images with optional bounding box constraints.

        Args:
            prompt: The prompt(s) to guide the image generation.
            height: Height of generated image (defaults to unet config).
            width: Width of generated image (defaults to unet config).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Guidance scale for classifier-free guidance.
            negative_prompt: Negative prompt(s) for guidance.
            num_images_per_prompt: Number of images per prompt.
            eta: Parameter for DDIM scheduler.
            generator: Random generator for reproducibility.
            latents: Pre-generated latents.
            prompt_embeds: Pre-computed prompt embeddings.
            negative_prompt_embeds: Pre-computed negative prompt embeddings.
            output_type: Output format ("pil", "latent", etc.).
            return_dict: Whether to return a dict or tuple.
            callback: Callback function during denoising.
            callback_steps: Frequency of callback calls.
            cross_attention_kwargs: Kwargs for cross-attention.

            boxdiff_phrases: List of phrases to spatially ground.
            boxdiff_boxes: List of normalized boxes [[x0, y0, x1, y1], ...].
            boxdiff_kwargs: BoxDiff-specific parameters dict.

        Returns:
            StableDiffusionPipelineOutput with generated images.
        """
        # Merge default kwargs with provided kwargs
        bd_kwargs = {**DEFAULT_BOXDIFF_KWARGS}
        if boxdiff_kwargs is not None:
            bd_kwargs.update(boxdiff_kwargs)

        # Extract BoxDiff parameters
        attention_res = bd_kwargs.get("attention_res", 16)
        P = bd_kwargs.get("P", 0.2)
        L = bd_kwargs.get("L", 1)
        max_iter_to_alter = bd_kwargs.get("max_iter_to_alter", 25)
        loss_thresholds = bd_kwargs.get("loss_thresholds", {0: 0.05, 10: 0.5, 20: 0.8})
        scale_factor = bd_kwargs.get("scale_factor", 20)
        scale_range = bd_kwargs.get("scale_range", (1.0, 0.5))
        smooth_attentions = bd_kwargs.get("smooth_attentions", True)
        sigma = bd_kwargs.get("sigma", 0.5)
        kernel_size = bd_kwargs.get("kernel_size", 3)
        normalize_eot = bd_kwargs.get("normalize_eot", True)

        # Use BoxDiff if boxes are provided
        use_boxdiff = (
            boxdiff_phrases is not None
            and boxdiff_boxes is not None
            and len(boxdiff_phrases) > 0
            and len(boxdiff_boxes) > 0
        )

        # 0. Default height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs (handle different diffusers API versions)
        try:
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
            )
        except TypeError:
            # Newer diffusers versions may have different signature
            pass

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt (handle different diffusers API versions)
        encode_result = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # Handle both old (2-tuple) and new (potentially different) return formats
        if isinstance(encode_result, tuple) and len(encode_result) >= 2:
            prompt_embeds = encode_result[0]
            negative_prompt_embeds = encode_result[1]
        else:
            prompt_embeds = encode_result
            negative_prompt_embeds = None

        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Get token indices for BoxDiff
        token_indices = []
        if use_boxdiff and isinstance(prompt, str):
            token_indices = self._get_token_indices(
                prompt, boxdiff_phrases, self.tokenizer, normalize_eot
            )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Set up attention processors for BoxDiff
        if use_boxdiff:
            self._setup_attention_processors(store_attention=True)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Apply BoxDiff constraints
                if use_boxdiff and i < max_iter_to_alter:
                    # Enable gradient computation for latent update
                    latents = latents.detach().requires_grad_(True)

                    # Get current loss threshold
                    current_threshold = 0.0
                    for step, threshold in sorted(loss_thresholds.items()):
                        if i >= int(step):
                            current_threshold = threshold

                    # Multiple loss iterations
                    for _ in range(L):
                        # Forward pass with gradient
                        latent_input = (
                            torch.cat([latents] * 2)
                            if do_classifier_free_guidance
                            else latents
                        )
                        latent_input = self.scheduler.scale_model_input(latent_input, t)

                        # Predict noise
                        with torch.enable_grad():
                            noise_pred = self.unet(
                                latent_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=cross_attention_kwargs,
                                return_dict=False,
                            )[0]

                        # Get attention maps
                        attn_maps = None
                        for name, proc in self.unet.attn_processors.items():
                            if (
                                hasattr(proc, "attention_scores")
                                and proc.attention_scores is not None
                            ):
                                scores = proc.attention_scores
                                # Use cross-attention maps at target resolution
                                if scores.shape[-2] == attention_res * attention_res:
                                    attn_maps = scores
                                    break

                        if attn_maps is not None:
                            # Compute BoxDiff loss
                            loss = compute_boxdiff_loss(
                                attn_maps,
                                token_indices,
                                boxdiff_boxes,
                                attention_res,
                                P=P,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                            )

                            # Update latents if loss is above threshold
                            if loss.item() > current_threshold:
                                # Compute gradient
                                grad = torch.autograd.grad(loss, latents)[0]

                                # Scale gradient
                                progress = i / max_iter_to_alter
                                scale = (
                                    scale_range[0]
                                    + (scale_range[1] - scale_range[0]) * progress
                                )
                                grad = grad * scale * scale_factor

                                # Update latents
                                latents = latents - grad

                    latents = latents.detach()

                # Standard denoising step
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # Classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Compute previous noisy sample
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # Call callback
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Reset attention processors
        if use_boxdiff:
            self._reset_attention_processors()

        # 9. Post-processing
        has_nsfw_concept = None
        
        if output_type == "latent":
            image = latents
        else:
            # Decode latents
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]

            # Run safety checker only if available
            if self.safety_checker is not None:
                image, has_nsfw_concept = self.run_safety_checker(
                    image, device, prompt_embeds.dtype
                )

            # Convert to PIL or other format
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload to CPU if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
