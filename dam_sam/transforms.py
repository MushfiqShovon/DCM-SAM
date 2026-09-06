import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image

# facebook/sam-vit-huge expects a square RGB input, normalized with ImageNet stats.
SAM_IMAGE_SIZE = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path: str, size: int = SAM_IMAGE_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return torch.from_numpy(arr)


def load_mask(path: str, size: int = SAM_IMAGE_SIZE, threshold: int = 127) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    arr = (np.asarray(mask) > threshold).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def get_valid_region(gray_arr: np.ndarray) -> np.ndarray:
    """Boolean mask of the imaged specimen (vs. surrounding background/air) in a raw
    grayscale XCT slice, via Otsu thresholding plus a fitted enclosing circle. AM/XCT
    scans are circular cross sections, so predictions and ground truth outside this disk
    are background artifacts, not true negatives, and should be excluded from metrics --
    matches the original DAM-SAM/XCT-SAM evaluation protocol exactly."""
    import cv2
    from scipy import ndimage as ndi

    otsu_val, _ = cv2.threshold(gray_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = otsu_val if otsu_val > 1 else gray_arr.max() * 0.10
    mask = gray_arr > threshold
    mask_filled = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask_filled = ndi.binary_fill_holes(mask_filled)
    contours, _ = cv2.findContours(mask_filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    valid = np.zeros_like(mask_filled, dtype=np.uint8)
    cv2.circle(valid, (int(x), int(y)), int(radius), 1, -1)
    return valid.astype(bool)


def joint_augment(
    image: torch.Tensor, pore_mask: torch.Tensor, incl_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random h-flip / v-flip / 90-degree rotation, applied identically to the image and
    both defect masks so they stay pixel-aligned."""
    if random.random() < 0.5:
        image, pore_mask, incl_mask = (torch.flip(t, dims=[-1]) for t in (image, pore_mask, incl_mask))
    if random.random() < 0.5:
        image, pore_mask, incl_mask = (torch.flip(t, dims=[-2]) for t in (image, pore_mask, incl_mask))
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])  # 90 / 180 / 270 degrees
        image, pore_mask, incl_mask = (torch.rot90(t, k, dims=[-2, -1]) for t in (image, pore_mask, incl_mask))
    return image, pore_mask, incl_mask
