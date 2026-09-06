# Trimmed from autogluon.multimodal.models.utils (create_adaptation / inject_adaptation_to_linear_layer),
# keeping only the conv_lora path used by DAM-SAM.
import re
from typing import List, Optional

import torch.nn as nn

from .adaptation_layers import ConvLoRALinear

# Matches the original DAM-SAM/XCT-SAM preset: Conv-LoRA is only injected into the vision
# encoder's attention blocks, and only into the "q"/"v" children. HF SamVisionAttention uses
# a single combined `qkv` Linear, so filter="q" (re.match, so prefix match) hits `qkv` while
# leaving `proj` (the output projection) untouched.
SAM_VISION_ENCODER_MODULE_FILTER = [r".*vision_encoder.*attn"]
SAM_VISION_ENCODER_LINEAR_FILTER = ["q", "v"]


def create_conv_lora_adaptation(
    layer: nn.Linear,
    lora_r: int,
    lora_alpha: int,
    conv_lora_expert_num: int,
) -> ConvLoRALinear:
    return ConvLoRALinear(
        layer.in_features,
        layer.out_features,
        r=lora_r,
        lora_alpha=lora_alpha,
        merge_weights=False,
        conv_lora_expert_num=conv_lora_expert_num,
    )


def inject_conv_lora(
    model: nn.Module,
    lora_r: int,
    lora_alpha: int,
    conv_lora_expert_num: int,
    module_filter: Optional[List[str]] = None,
    filter: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replace matching nn.Linear layers in-place with ConvLoRALinear.

    Parameters
    ----------
    model
        A PyTorch model (mutated in place).
    lora_r, lora_alpha
        Low-rank decomposition rank / scaling factor.
    conv_lora_expert_num
        Number of multi-scale conv experts in the MoE-Conv gate.
    module_filter
        Regex list; only modules whose name matches (re.match) one of these are considered.
        Defaults to SAM_VISION_ENCODER_MODULE_FILTER.
    filter
        Regex list; only direct-child Linear layers whose name matches (re.match) one of
        these are replaced. Defaults to SAM_VISION_ENCODER_LINEAR_FILTER.
    """
    module_filter = module_filter if module_filter is not None else SAM_VISION_ENCODER_MODULE_FILTER
    filter = filter if filter is not None else SAM_VISION_ENCODER_LINEAR_FILTER

    for m_name, module in dict(model.named_modules()).items():
        if not any(re.match(pat, m_name) for pat in module_filter):
            continue
        for c_name, layer in dict(module.named_children()).items():
            if not any(re.match(pat, c_name) for pat in filter):
                continue
            assert isinstance(layer, nn.Linear), f"Conv-LoRA can only replace nn.Linear, got {type(layer)} at {m_name}.{c_name}"
            adapted = create_conv_lora_adaptation(layer, lora_r, lora_alpha, conv_lora_expert_num)
            adapted.weight = layer.weight
            adapted.bias = layer.bias
            setattr(module, c_name, adapted)

    return model
