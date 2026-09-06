from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_inject import inject_conv_lora
from .modeling_sam_conv_lora import SamModel


@dataclass
class DamSAMOutput:
    pred_masks: torch.Tensor  # (B, 1, image_size, image_size) logits, full resolution
    moe_loss: torch.Tensor  # scalar, Conv-LoRA load-balancing loss


class DamSAMHead(nn.Module):
    """
    One promptless segmentation head: a frozen SAM ViT-H encoder adapted with Conv-LoRA (a
    mixture of multi-scale conv experts inside the vision encoder's qkv LoRA branch), plus a
    fully trainable mask decoder. Unlike BaT-SAM, there is no box regressor and no prompt at
    all -- the model is called with only pixel_values, so the prompt encoder's sparse and
    dense embeddings both fall back to their learned "no prompt given" defaults, and the
    decoder must learn to segment the target defect class directly from the adapted image
    embedding alone. This matches the promptless "automatic" design of the original
    DAM-SAM/XCT-SAM training scripts.

    DAM-SAM trains one `DamSAMHead` per defect class (pore, inclusion) rather than sharing a
    single adapter and decoder across classes: each gets its own Conv-LoRA weights and its
    own mask decoder, with completely separate forward/backward passes and optimizers (see
    scripts/train.py). This is the opposite tradeoff from BaT-SAM, which unified all tumor
    types behind one adapter using a self-prompting box regressor to disambiguate location
    and class; here, pore and inclusion defects differ enough in scale, density, and false
    positive risk that dedicated capacity per class was worth the ~2x parameter cost (still
    under 1% of SAM's total per head).

    Trainable: Conv-LoRA params (`lora_*`) in the vision encoder, and the full mask decoder
    including `iou_prediction_head`. Frozen: everything else, including the base ViT-H
    weights and the prompt encoder.
    """

    def __init__(
        self,
        checkpoint_name: str = "facebook/sam-vit-base",
        lora_r: int = 2,
        # scaling = lora_alpha / lora_r = 4.0, matching XCT-SAM's own effective scaling at
        # r=2 (XCT-SAM's scripts only ever override AutoGluon's optim.lora.r, leaving
        # optim.lora.alpha at its config default of 8, so alpha=8 here reproduces that same
        # r=2/alpha=8 setting exactly, rather than the untested alpha=32 the original
        # DAM-SAM_back script defaulted to, scaling=16, with no found rationale for it).
        lora_alpha: int = 8,
        conv_lora_expert_num: int = 8,
        image_size: int = 1024,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.sam = SamModel.from_pretrained(checkpoint_name)
        self._adapt_input_resolution(self.sam, image_size)
        inject_conv_lora(
            self.sam,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            conv_lora_expert_num=conv_lora_expert_num,
        )
        self.image_size = image_size
        self.sam.vision_encoder.gradient_checkpointing = gradient_checkpointing
        self._set_trainable()


    @staticmethod
    def _adapt_input_resolution(sam, image_size: int) -> None:
        """Retarget a pretrained SAM vision encoder to a different square input resolution.

        SAM ships at 1024x1024 and hard-rejects anything else: SamPatchEmbeddings validates
        the input against config.image_size, and pos_embed is a fixed (1, 64, 64, C) grid.
        Both are updated here, along with the prompt encoder's patch-grid size, which sets
        the shape of the dense "no prompt" embedding added to the image embedding in the
        decoder. Relative-position tables need nothing -- get_rel_pos() interpolates them
        unconditionally at every call.

        Attention memory scales with tokens^2, i.e. the FOURTH power of the linear input
        size, so 1024 -> 512 shrinks the ViT-H attention tensor from 512 MB to 32 MB. The
        resampled pos_embed is NOT pretrained at the new resolution, so a retargeted model
        must be fine-tuned; it is not a zero-shot change.
        """
        enc = sam.vision_encoder
        if image_size == enc.config.image_size:
            return
        patch = enc.config.patch_size
        if image_size % patch != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch}")
        grid = image_size // patch

        enc.patch_embed.image_size = (image_size, image_size)
        enc.config.image_size = image_size
        sam.config.vision_config.image_size = image_size
        sam.config.prompt_encoder_config.image_embedding_size = grid
        if hasattr(sam, "prompt_encoder"):
            sam.prompt_encoder.image_embedding_size = (grid, grid)

        if getattr(enc, "pos_embed", None) is not None:
            pe = enc.pos_embed.data
            dtype = pe.dtype
            pe = F.interpolate(pe.permute(0, 3, 1, 2).float(), size=(grid, grid),
                               mode="bicubic", align_corners=False)
            enc.pos_embed = nn.Parameter(pe.permute(0, 2, 3, 1).contiguous().to(dtype))

    def _set_trainable(self):
        for name, p in self.sam.named_parameters():
            trainable = "lora_" in name or name.startswith("mask_decoder")
            p.requires_grad_(trainable)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def trainable_state_dict(self):
        """Only the Conv-LoRA + mask decoder params (~4M), not the ~627M frozen ViT-H
        backbone, which reloads identically from `checkpoint_name` every time a DamSAMHead
        is constructed, so persisting it is pure waste."""
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in trainable_names}

    def forward(self, pixel_values: torch.Tensor) -> DamSAMOutput:
        """pixel_values: (B, 3, image_size, image_size), ImageNet-normalized. No prompt of
        any kind is passed to SAM; segmentation is fully automatic."""
        sam_out = self.sam(
            pixel_values,
            multimask_output=False,
            output_moe_loss=True,
            return_dict=True,
        )
        low_res_masks = sam_out.pred_masks[:, 0]  # (B, 1, h, w)
        pred_masks = F.interpolate(
            low_res_masks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        return DamSAMOutput(pred_masks=pred_masks, moe_loss=sam_out.vision_moe_loss)
