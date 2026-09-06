import os

import matplotlib

matplotlib.use("Agg")  # headless: never calls plt.show(), only savefig
import matplotlib.pyplot as plt
import numpy as np
import torch

from .transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """image: (3, H, W), ImageNet-normalized RGB. Returns an (H, W, 3) array in [0, 1]."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    arr = image.detach().cpu().float().numpy() * std + mean
    return np.clip(arr, 0.0, 1.0).transpose(1, 2, 0)


def save_prediction_panel(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask_logits: torch.Tensor,
    name: str,
    output_dir: str,
    class_tag: str,
    iou: float = None,
    dice: float = None,
    threshold: float = 0.5,
):
    """Saves a 3-panel [input | ground truth | prediction] PNG.

    image: (3, H, W). gt_mask: (1, H, W) in {0, 1}. pred_mask_logits: (1, H, W) raw logits
    (sigmoid + threshold applied here for display, matching the metric's own threshold).
    class_tag: "pore" or "incl", included in the filename and panel titles.
    """
    os.makedirs(output_dir, exist_ok=True)

    img_arr = denormalize_image(image)
    gt_arr = gt_mask[0].detach().cpu().float().numpy()
    pred_arr = (torch.sigmoid(pred_mask_logits[0]).detach().cpu().float().numpy() > threshold).astype(np.float32)

    title_bits = []
    if iou is not None:
        title_bits.append(f"IoU={iou:.4f}")
    if dice is not None:
        title_bits.append(f"Dice={dice:.4f}")
    pred_title = f"{class_tag.capitalize()} prediction" + (f"  ({'  '.join(title_bits)})" if title_bits else "")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_arr)
    for ax, arr, title in zip(axes[1:], [gt_arr, pred_arr], [f"{class_tag.capitalize()} ground truth", pred_title]):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    axes[0].set_title("Input")
    axes[0].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{class_tag}_{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
