import random
import re
from typing import List, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_VIT_BLOCK_RE = re.compile(r"layers\.(\d+)\.")


def resolve_devices(pore_device: str = None, incl_device: str = None):
    """If both pore_device and incl_device are left unset (None), auto head-parallel split
    across GPUs: cuda:0/cuda:1 when a second GPU is visible, else both on cuda:0 (or cpu).
    Either one being set explicitly disables auto-splitting for the other, so a single
    incl_device='cuda:1' still works without also having to spell out pore_device."""
    n_gpus = torch.cuda.device_count()
    if pore_device is None and incl_device is None:
        if n_gpus >= 2:
            return torch.device("cuda:0"), torch.device("cuda:1")
        return torch.device("cuda:0" if n_gpus == 1 else "cpu"), torch.device("cuda:0" if n_gpus == 1 else "cpu")
    default = "cuda:0" if n_gpus >= 1 else "cpu"
    return torch.device(pore_device or default), torch.device(incl_device or default)


def seed_everything(seed: int = 42686693):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


@torch.no_grad()
def mask_iou_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """logits, targets: (B, 1, H, W). Returns per-batch mean IoU and Dice."""
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = (targets > 0.5).float()
    preds = preds.flatten(1)
    targets = targets.flatten(1)

    intersection = (preds * targets).sum(-1)
    union = (preds + targets).clamp(0, 1).sum(-1)
    iou = (intersection + eps) / (union + eps)

    dice = (2 * intersection + eps) / (preds.sum(-1) + targets.sum(-1) + eps)
    return iou.mean().item(), dice.mean().item()


@torch.no_grad()
def tolerance_f1(logits: torch.Tensor, targets: torch.Tensor, tolerance: int = 5, threshold: float = 0.5):
    """logits, targets: (1, H, W). Tolerance-based precision/recall/F1: a predicted pixel is
    a true positive if it falls within `tolerance` pixels of any labeled pixel, forgiving
    small boundary offsets that sink strict IoU/Dice on small, sparse defects."""
    pred = (torch.sigmoid(logits) > threshold)[0].detach().cpu().numpy().astype(bool)
    gt = (targets > 0.5)[0].detach().cpu().numpy().astype(bool)
    return tolerance_f1_masks(pred, gt, tolerance)


def tolerance_f1_masks(pred: np.ndarray, gt: np.ndarray, tolerance: int = 5):
    """Same metric as `tolerance_f1`, but for a pair of already-binarized boolean masks
    (e.g. after native-resolution resizing, morphological cleanup, and valid-region masking
    at eval time, where there's no single-resolution logits tensor to threshold)."""
    from scipy import ndimage as ndi

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0, 1.0, 1.0
    if gt.sum() == 0 or pred.sum() == 0:
        return 0.0, 0.0, 0.0

    struct = ndi.generate_binary_structure(2, 1)
    dilated_gt = ndi.binary_dilation(gt, structure=struct, iterations=tolerance)
    dilated_pred = ndi.binary_dilation(pred, structure=struct, iterations=tolerance)

    tp = np.logical_and(pred, dilated_gt).sum()
    fp = np.logical_and(pred, ~dilated_gt).sum()
    fn = np.logical_and(gt, ~dilated_pred).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(precision), float(recall), float(f1)


def get_layerwise_param_groups(model, base_lr: float, lr_decay: float, num_blocks: Optional[int] = None) -> List[dict]:
    """Exponentially decayed LR for earlier ViT blocks: block 0 (closest to the patch
    embedding, most fragile under fine-tuning) gets base_lr * lr_decay^(num_blocks - 1),
    the deepest block and the mask decoder get base_lr. Only ever touches parameters that
    already have requires_grad=True (the frozen backbone is untouched either way).

    num_blocks defaults to None, which auto-detects the encoder depth from the highest
    "layers.N." block index actually present among the model's trainable parameters,
    rather than assuming ViT-H's 32 blocks -- this is what makes the same call site work
    unchanged for ViT-B (12 blocks) or ViT-L (24 blocks) after swapping --checkpoint_name."""
    by_block: dict = {}
    decoder = []
    other = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        m = _VIT_BLOCK_RE.search(name)
        if m:
            by_block.setdefault(int(m.group(1)), []).append(param)
        elif "mask_decoder" in name:
            decoder.append(param)
        else:
            other.append(param)

    if num_blocks is None:
        if not by_block:
            raise ValueError(
                "get_layerwise_param_groups found no trainable 'layers.N.' parameters to "
                "auto-detect encoder depth from -- pass num_blocks explicitly."
            )
        num_blocks = max(by_block) + 1

    groups = []
    if decoder:
        groups.append({"params": decoder, "lr": base_lr})
    for blk in sorted(by_block):
        groups.append({"params": by_block[blk], "lr": base_lr * (lr_decay ** (num_blocks - 1 - blk))})
    if other:
        groups.append({"params": other, "lr": base_lr * (lr_decay ** num_blocks)})
    return groups


def make_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Step-level linear warmup (1% to 100% LR) then cosine decay to 0. Must be stepped once
    per optimizer step, not per epoch."""
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(1, warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def save_joint_checkpoint(path: str, pore_model, incl_model, epoch: int = 0, extra: dict = None):
    """Saves only the trainable delta of each head (Conv-LoRA + mask decoder, ~4M params
    each), not the frozen ~627M ViT-H backbone shared by both -- see DamSAMHead's
    trainable_state_dict. Keys are prefixed pore.*/incl.* in a single file, so a trained
    model ships as one ~32MB checkpoint instead of two."""
    state = {
        "pore": pore_model.trainable_state_dict(),
        "incl": incl_model.trainable_state_dict(),
        "epoch": epoch,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_joint_checkpoint(path: str, pore_model, incl_model, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    for tag, model in (("pore", pore_model), ("incl", incl_model)):
        missing, unexpected = model.load_state_dict(state[tag], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in {tag} checkpoint (not in model): {unexpected}")
    return state
