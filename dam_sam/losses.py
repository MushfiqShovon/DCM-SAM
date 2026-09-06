import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """logits, targets: (B, 1, H, W). targets in {0, 1}."""
    probs = torch.sigmoid(logits.float()).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(-1)
    union = probs.sum(-1) + targets.sum(-1)
    dice = (2 * intersection + eps) / (union + eps)
    return (1 - dice).mean()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss, logits/targets: (B, 1, H, W)."""
    # p_t = exp(-bce) is bounded in (0, 1] in exact arithmetic, so base = 1 - p_t is bounded
    # in [0, 1). The problem is exactly 0, which real predictions do reach at full float32
    # precision (a perfectly confident correct pixel gives bce == 0.0 exactly, so p_t == 1.0
    # and base == 0.0 exactly, not an edge case, a routine occurrence on any easy/background
    # pixel). torch.pow's backward computes the base's gradient as
    # grad_output * gamma * result / base -- a division-based formula, not the direct
    # calculus derivative -- so an exact-zero base produces 0/0 = NaN in the gradient even
    # though the true derivative there is a well-defined 0 and the forward value is
    # perfectly finite. Confirmed directly: a gradient hook on `base` showed base == 0.0 and
    # bce == 0.0 at every single non-finite-gradient element, with base/bce/p_t all
    # otherwise finite -- clamping the base to exactly 0 (an earlier, incorrect fix) does not
    # avoid this, since 0 is the singularity itself, not just its neighborhood. This NaN
    # gradient then passes straight through gradient clipping (which cannot repair a
    # non-finite value) into optimizer.step(), permanently corrupting the model weights --
    # which is what later actually surfaced, several steps on, as an unrelated-looking crash
    # deep in Conv-LoRA's MoE gating.
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    p_t = torch.exp(-bce)
    base = (1 - p_t).clamp(min=1e-7)
    focal = alpha * base**gamma * bce
    return focal.flatten(1).mean(-1).mean()


class DiceFocalLoss(nn.Module):
    """Equal-weighted Dice + focal loss. Handles the extreme class imbalance of sparse
    defect masks: Dice normalizes by foreground size, focal down-weights easy background
    pixels so gradient concentrates on hard/rare defect pixels."""

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * dice_loss(logits, targets) + self.focal_weight * focal_loss(
            logits, targets, self.alpha, self.gamma
        )


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for binary segmentation of small, sparse defects.

    Generalizes Dice via the Tversky index, which separately weights false positives
    (alpha) and false negatives (beta). beta > alpha penalizes missed defects more than
    spurious ones. The focal exponent gamma further down-weights easy examples.

    Tversky index  TI = TP / (TP + alpha*FP + beta*FN)
    Loss           L  = (1 - TI) ** gamma

    Reference: Abraham & Khan, "A Novel Focal Tversky Loss Function With Improved
    Attention U-Net for Lesion Segmentation", ISBI 2019. https://arxiv.org/abs/1810.07842
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # Cast to float32 under AMP: float16 can push tversky slightly above 1, making
        # (1 - tversky)^gamma NaN for gamma < 1.
        probs = torch.sigmoid(logits.float()).flatten(1)
        targets = targets.flatten(1)

        tp = (probs * targets).sum(-1)
        fp = (probs * (1.0 - targets)).sum(-1)
        fn = ((1.0 - probs) * targets).sum(-1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Clamp the base away from 0: d/dx[x^gamma] at x=0 is inf for gamma < 1, which
        # produces NaN once tversky is close to 1 (near-empty target masks).
        base = (1.0 - tversky.clamp(0.0, 1.0)).clamp(min=1e-7)
        return (base**self.gamma).mean()


class BCEDiceLoss(nn.Module):
    """BCE-with-pos_weight + soft Dice, for ultra-sparse targets where a handful of
    positive pixels would otherwise be swamped by background in a plain BCE."""

    def __init__(self, pos_weight: float = 100.0, dice_weight: float = 0.5):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight.to(logits.device))
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        dice = 1.0 - (2.0 * tp + 1.0) / (probs.sum() + targets.sum() + 1.0)
        return bce + self.dice_weight * dice


def build_loss(name: str, **kwargs) -> nn.Module:
    if name == "dice_focal_loss":
        return DiceFocalLoss(**{k: v for k, v in kwargs.items() if k in ("dice_weight", "focal_weight", "alpha", "gamma")})
    if name == "focal_tversky_loss":
        return FocalTverskyLoss(**{k: v for k, v in kwargs.items() if k in ("alpha", "beta", "gamma", "smooth")})
    if name == "bce_dice_loss":
        return BCEDiceLoss(**{k: v for k, v in kwargs.items() if k in ("pos_weight", "dice_weight")})
    raise ValueError(f"Unknown loss: {name!r}")


def per_image_loss(loss_fn: nn.Module, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Averages the loss computed independently on each image in the batch, rather than
    over the whole batch at once. With a small batch size and a heavily oversampled sparse
    class, one dense image can otherwise dominate a mean computed across the flattened
    batch; per-image averaging keeps every sample's contribution equal regardless of how
    much foreground it contains."""
    total = sum(loss_fn(logits[i : i + 1], targets[i : i + 1]) for i in range(logits.shape[0]))
    return total / logits.shape[0]
