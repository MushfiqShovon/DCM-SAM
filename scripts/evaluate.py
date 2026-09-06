"""
evaluate.py — DAM-SAM joint checkpoint evaluation.

Accepts --split as  <dataset>/<split_prefix>  (e.g. gan-generated/test1, nist/test2).
Reads  datasets/<dataset>/<split_prefix>_class2.csv  for pore.
Reads  datasets/<dataset>/<split_prefix>_class3.csv  for inclusion (skipped if absent or
if --pore_only is set).
Outputs go to  outputs/<dataset>/<split_prefix>/  to avoid overwriting across runs.

Panel layout
  pore + inclusion : Input | Pore GT | Incl GT | Pore Pred | Incl Pred   (5 panels)
  pore only        : Input | Pore GT | Pore Pred                          (3 panels)

Usage examples
  python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt \\
      --split gan-generated/test1
  python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt \\
      --split gan-generated/test4 --pore_only
  python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt \\
      --split nist/test2
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure as sk_exposure
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dam_sam import DamSAMHead
from dam_sam.transforms import IMAGENET_MEAN, IMAGENET_STD, SAM_IMAGE_SIZE, get_valid_region
from dam_sam.utils import load_joint_checkpoint, resolve_devices, tolerance_f1_masks


# ─────────────────────────── CLI ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a DAM-SAM joint checkpoint on a given dataset/split."
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the joint checkpoint file (*.pt / *.ckpt).",
    )
    p.add_argument(
        "--split", type=str, required=True,
        help="'<dataset>/<split_prefix>', e.g. 'gan-generated/test1' or 'nist/test2'. "
             "The script reads datasets/<dataset>/<split_prefix>_class2.csv (pore) "
             "and _class3.csv (inclusion, if present and --pore_only not set).",
    )
    p.add_argument(
        "--datasets_root", type=str, default="datasets",
        help="Root directory containing dataset subdirectories (default: datasets/).",
    )
    p.add_argument(
        "--outputs_root", type=str, default="outputs",
        help="Root directory for outputs. Results go to <outputs_root>/<dataset>/<split>/.",
    )
    p.add_argument("--checkpoint_name", type=str, default="facebook/sam-vit-base")
    p.add_argument("--image_size", type=int, default=SAM_IMAGE_SIZE,
                   help="Must match the value the checkpoint was trained with.")
    p.add_argument("--lora_r",     type=int,   default=2)
    p.add_argument("--lora_alpha", type=int,   default=8,
                   help="Must match the checkpoint's training --lora_alpha.")
    p.add_argument("--expert_num", type=int,   default=8)
    p.add_argument("--pore_device", type=str,  default=None,
                   help="Defaults to cuda:0, or auto cuda:0/cuda:1 split if two GPUs visible.")
    p.add_argument("--incl_device", type=str,  default=None)
    # ── thresholds (per-head) ─────────────────────────────────────────────────
    p.add_argument("--threshold_pore", type=float, default=0.5,
                   help="Sigmoid threshold for pore head (lower → more sensitive).")
    p.add_argument("--threshold_incl", type=float, default=0.5,
                   help="Sigmoid threshold for inclusion head.")
    p.add_argument("--morph_size_pore", type=int, default=1,
                   help="Binary-opening kernel for pore mask (0 = disabled). "
                        "Set 0 for tiny sparse pores to avoid erasing them.")
    p.add_argument("--morph_size_incl", type=int, default=1)

    # ── preprocessing enhancements ────────────────────────────────────────────
    p.add_argument("--apply_clahe", action="store_true", default=False,
                   help="Apply CLAHE contrast enhancement before encoding "
                        "(strongly recommended for dark / out-of-distribution images).")
    p.add_argument("--clahe_clip_limit", type=float, default=0.03,
                   help="CLAHE clip limit [0–1]. Higher = more aggressive contrast boost.")
    p.add_argument("--clahe_grid_size", type=int, default=8,
                   help="CLAHE tile grid size (applied to each NxN tile independently).")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Gamma correction applied after CLAHE (γ<1 brightens, γ>1 darkens). "
                        "Useful for very dark scans (try 0.6–0.8).")
    p.add_argument("--tta", action="store_true", default=False,
                   help="Test-time augmentation: average predictions over original + "
                        "horizontal flip + vertical flip (reduces spatial bias artifacts).")

    # ── connected-component size filtering ────────────────────────────────────
    p.add_argument("--min_blob_pore", type=int, default=0,
                   help="Discard pore components with fewer pixels than this (removes 1-px noise).")
    p.add_argument("--max_blob_pore", type=int, default=0,
                   help="Discard pore components with more pixels than this (removes large blobs). "
                        "0 = disabled. For tiny sparse pores (e.g. Ti64 test1) try 200–500.")
    p.add_argument("--min_blob_incl", type=int, default=0,
                   help="Discard inclusion components smaller than this.")
    p.add_argument("--max_blob_incl", type=int, default=0,
                   help="Discard inclusion components larger than this. 0 = disabled.")

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--use_valid_region", action="store_true", default=True,
                   help="Mask predictions/GT to Otsu-derived specimen disk before scoring.")
    p.add_argument("--no_valid_region",  dest="use_valid_region", action="store_false")
    p.add_argument("--tol_f1_tolerance", type=int, default=5)
    p.add_argument("--pore_only", action="store_true", default=False,
                   help="Skip inclusion evaluation even if a class3 CSV exists "
                        "(use for GAN test4/test5 which have no inclusion labels).")
    p.add_argument("--save_images", action="store_true", default=True,
                   help="Save per-image panel figures (default: on).")
    p.add_argument("--no_save_images", dest="save_images", action="store_false")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


# ─────────────────────────── image helpers ────────────────────────────────────

def preprocess(img_pil: Image.Image, args) -> torch.Tensor:
    """
    Convert PIL image to SAM input tensor, with optional enhancements:
      1. CLAHE  – adaptive histogram equalisation (helps dark / OOD images)
      2. Gamma  – power-law brightness correction (γ < 1 brightens)
    Both operate on the grayscale channel before RGB conversion so that
    SAM's ImageNet normalisation sees a contrast-enhanced image.
    """
    gray = np.array(img_pil.convert("L"))

    if args.apply_clahe:
        # equalize_adapthist returns float32 in [0, 1]
        gray = (sk_exposure.equalize_adapthist(
            gray,
            kernel_size=args.clahe_grid_size,
            clip_limit=args.clahe_clip_limit,
        ) * 255).astype(np.uint8)

    if args.gamma != 1.0:
        gray = (255.0 * np.power(gray / 255.0, args.gamma)).clip(0, 255).astype(np.uint8)

    img_proc = Image.fromarray(gray).convert("RGB").resize(
        (getattr(args, "image_size", SAM_IMAGE_SIZE),
         getattr(args, "image_size", SAM_IMAGE_SIZE)), Image.BILINEAR
    )
    arr  = np.asarray(img_proc, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
    return torch.from_numpy((arr - mean) / std).unsqueeze(0)


@torch.no_grad()
def predict_mask(model, device, img_tensor, morph_size, threshold, orig_h, orig_w, tta=False):
    """
    Run one head and return a binary mask at native resolution.
    With tta=True, predictions are averaged over the original image plus its
    horizontal and vertical flips before thresholding, reducing spatial bias.
    """
    variants = [(img_tensor, False, False)]
    if tta:
        variants += [
            (img_tensor.flip(-1), True,  False),  # horizontal flip
            (img_tensor.flip(-2), False, True),   # vertical flip
        ]

    probs = []
    for tensor, hflip, vflip in variants:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(tensor.to(device))
        p = torch.sigmoid(out.pred_masks.float())
        if hflip:
            p = p.flip(-1)
        if vflip:
            p = p.flip(-2)
        probs.append(F.interpolate(p, size=(orig_h, orig_w), mode="bilinear", align_corners=False))

    avg_prob = torch.stack(probs).mean(0)
    binary = (avg_prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    if morph_size > 0:
        binary = ndi.binary_opening(binary, np.ones((morph_size, morph_size))).astype(np.uint8)
    return binary


def filter_components(binary, min_pixels=0, max_pixels=0):
    """
    Remove connected components that are too small (noise) or too large (blobs).
    max_pixels=0 means no upper limit.
    Runs only when at least one bound is active, so default args are free.
    """
    if min_pixels == 0 and max_pixels == 0:
        return binary
    labeled, n = ndi.label(binary)
    if n == 0:
        return binary
    sizes = np.bincount(labeled.ravel())   # sizes[0] = background
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    if min_pixels > 0:
        keep[1:] &= (sizes[1:] >= min_pixels)
    if max_pixels > 0:
        keep[1:] &= (sizes[1:] <= max_pixels)
    return keep[labeled].astype(np.uint8)


# ─────────────────────────── panel saving ─────────────────────────────────────

_FONT   = {"fontname": "Times New Roman", "fontsize": 13}
_YELLOW = np.array([255, 255,   0], dtype=np.uint8)   # pore colour
_RED    = np.array([255,   0,   0], dtype=np.uint8)   # inclusion colour
_MARGIN = 5                                            # px white border around each panel


def _colorise(binary_mask, colour):
    """Return RGB image: defect pixels in `colour` on pure black background."""
    h, w = binary_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[binary_mask.astype(bool)] = colour
    return rgb


def _pad(arr, margin=_MARGIN):
    """Add `margin` pixels of white around a 2-D (gray) or 3-D (RGB) array."""
    if arr.ndim == 2:
        return np.pad(arr, margin, constant_values=255)
    return np.pad(arr, ((margin, margin), (margin, margin), (0, 0)), constant_values=255)


def save_panel(
    gray, pore_gt_raw, pore_binary,
    incl_gt_raw, incl_binary,
    fname, output_dir,
    p_iou, p_dice, p_f1,
    i_iou=None, i_dice=None, i_f1=None,
):
    """
    Panel order (pore + incl): Input | Pore GT | Pore Pred | Incl GT | Incl Pred
    Panel order (pore only)  : Input | Pore GT | Pore Pred
    GT/Pred masks are on a black background (yellow = pore, red = incl).
    Each panel has a 5 px white margin.
    """
    has_incl = incl_gt_raw is not None
    n = 5 if has_incl else 3

    panels = [
        (_pad(gray),                                          "Input"),
        (_pad(_colorise(pore_gt_raw,  _YELLOW)),              "Pore GT"),
        (_pad(_colorise(pore_binary,  _YELLOW)),
         f"Pore Pred    IoU: {p_iou:.4f}   Dice: {p_dice:.4f}   Tol-F1: {p_f1:.4f}"),
    ]
    if has_incl:
        panels += [
            (_pad(_colorise(incl_gt_raw, _RED)),              "Incl GT"),
            (_pad(_colorise(incl_binary, _RED)),
             f"Incl Pred    IoU: {i_iou:.4f}   Dice: {i_dice:.4f}   Tol-F1: {i_f1:.4f}"),
        ]

    fig = plt.figure(figsize=(5 * n, 5))
    for idx, (arr, title) in enumerate(panels, 1):
        ax = fig.add_subplot(1, n, idx)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        ax.set_title(title, **_FONT)
        ax.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()


# ─────────────────────────── iou / dice helpers ───────────────────────────────

def _iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(inter / union)


def _dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6))


# ─────────────────────────── evaluation loop ──────────────────────────────────

def evaluate(
    pore_model, pore_device,
    incl_model, incl_device,
    pore_df, incl_df,
    dataset_root, eval_incl, args,
):
    """
    Single image-level loop over pore_df; runs both heads per image so joint
    panels can be saved.  Returns (pore_metrics, incl_metrics_or_None).
    """
    pore_ious, pore_dices, pore_precs, pore_recs, pore_f1s = [], [], [], [], []
    pore_ious_pos = []
    p_empty_total = p_empty_correct = 0

    incl_ious, incl_dices, incl_precs, incl_recs, incl_f1s = [], [], [], [], []
    incl_ious_pos = []
    i_empty_total = i_empty_correct = 0

    incl_iter = iter(incl_df.iterrows()) if eval_incl else None

    for i, (_, pore_row) in enumerate(
        tqdm(pore_df.iterrows(), total=len(pore_df), desc="Inference", unit="img")
    ):
        img_path      = os.path.join(dataset_root, pore_row["image"])
        pore_lbl_path = os.path.join(dataset_root, pore_row["label"])

        img_pil  = Image.open(img_path)
        gray     = np.array(img_pil.convert("L"))
        orig_h, orig_w = gray.shape[:2]
        pv = preprocess(img_pil, args)

        # ── pore head ─────────────────────────────────────────────────────────
        pore_binary = predict_mask(
            pore_model, pore_device, pv,
            args.morph_size_pore, args.threshold_pore, orig_h, orig_w, tta=args.tta,
        )
        pore_binary = filter_components(pore_binary, args.min_blob_pore, args.max_blob_pore)
        pore_gt_raw = (np.array(Image.open(pore_lbl_path).convert("L")) > 0)

        if args.use_valid_region:
            valid     = get_valid_region(gray)
            pore_pred = np.where(valid, pore_binary, 0).astype(bool)
            pore_gt   = np.where(valid, pore_gt_raw, 0).astype(bool)
        else:
            pore_pred = pore_binary.astype(bool)
            pore_gt   = pore_gt_raw

        p_iou  = _iou(pore_pred, pore_gt)
        p_dice = _dice(pore_pred, pore_gt)
        p_prec, p_rec, p_f1 = tolerance_f1_masks(pore_pred, pore_gt, tolerance=args.tol_f1_tolerance)
        pore_ious.append(p_iou);  pore_dices.append(p_dice)
        pore_precs.append(p_prec); pore_recs.append(p_rec); pore_f1s.append(p_f1)
        if pore_gt.any():
            pore_ious_pos.append(p_iou)
        else:
            p_empty_total += 1
            if not pore_pred.any():
                p_empty_correct += 1

        # ── inclusion head ────────────────────────────────────────────────────
        incl_gt_raw = incl_binary = None
        i_iou = i_dice = i_f1 = None

        if eval_incl:
            _, incl_row   = next(incl_iter)
            incl_lbl_path = os.path.join(dataset_root, incl_row["label"])
            incl_binary   = predict_mask(
                incl_model, incl_device, pv,
                args.morph_size_incl, args.threshold_incl, orig_h, orig_w, tta=args.tta,
            )
            incl_binary = filter_components(incl_binary, args.min_blob_incl, args.max_blob_incl)
            incl_gt_raw = (np.array(Image.open(incl_lbl_path).convert("L")) > 0)

            if args.use_valid_region:
                incl_pred = np.where(valid, incl_binary, 0).astype(bool)
                incl_gt   = np.where(valid, incl_gt_raw, 0).astype(bool)
            else:
                incl_pred = incl_binary.astype(bool)
                incl_gt   = incl_gt_raw

            i_iou  = _iou(incl_pred, incl_gt)
            i_dice = _dice(incl_pred, incl_gt)
            i_prec, i_rec, i_f1 = tolerance_f1_masks(incl_pred, incl_gt, tolerance=args.tol_f1_tolerance)
            incl_ious.append(i_iou);  incl_dices.append(i_dice)
            incl_precs.append(i_prec); incl_recs.append(i_rec); incl_f1s.append(i_f1)
            if incl_gt.any():
                incl_ious_pos.append(i_iou)
            else:
                i_empty_total += 1
                if not incl_pred.any():
                    i_empty_correct += 1

        # ── save panel ────────────────────────────────────────────────────────
        if args.save_images:
            save_panel(
                gray, pore_gt_raw, pore_binary,
                incl_gt_raw, incl_binary,
                f"joint_{os.path.basename(img_path)}", args.output_dir,
                p_iou, p_dice, p_f1,
                i_iou, i_dice, i_f1,
            )

        if (i + 1) % args.log_every == 0:
            print(f"  [{i+1}/{len(pore_df)}]  pore IoU={np.mean(pore_ious):.4f}"
                  + (f"  incl IoU={np.mean(incl_ious):.4f}" if eval_incl else ""))

    # ── pack results ──────────────────────────────────────────────────────────
    p_tnr = p_empty_correct / p_empty_total if p_empty_total > 0 else 1.0
    pore_m = dict(
        ious=pore_ious, dices=pore_dices, f1s=pore_f1s,
        precs=pore_precs, recs=pore_recs, ious_pos=pore_ious_pos,
        n_empty_total=p_empty_total, n_empty_correct=p_empty_correct, tnr=p_tnr,
    )

    incl_m = None
    if eval_incl:
        i_tnr = i_empty_correct / i_empty_total if i_empty_total > 0 else 1.0
        incl_m = dict(
            ious=incl_ious, dices=incl_dices, f1s=incl_f1s,
            precs=incl_precs, recs=incl_recs, ious_pos=incl_ious_pos,
            n_empty_total=i_empty_total, n_empty_correct=i_empty_correct, tnr=i_tnr,
        )
    return pore_m, incl_m


# ─────────────────────────── reporting ───────────────────────────────────────

def _section_str(tag, split_csv, m):
    """Build a verbose metrics block matching the format summarize_metrics.py expects."""
    defect_line = (
        f"  Defect images     : {len(m['ious_pos'])}  IoU(pos):{np.mean(m['ious_pos']):.4f}\n"
        if m["ious_pos"] else ""
    )
    return (
        f"\n─── {tag} ── {split_csv} ─────────────────────────\n"
        f"  Mean IoU          : {np.mean(m['ious']):.4f}  ± {np.std(m['ious']):.4f}\n"
        f"  Mean Dice         : {np.mean(m['dices']):.4f}  ± {np.std(m['dices']):.4f}\n"
        f"  Mean Tol-F1 (5px) : {np.mean(m['f1s']):.4f}  "
        f"[P:{np.mean(m['precs']):.4f}  R:{np.mean(m['recs']):.4f}]\n"
        f"{defect_line}"
        f"  Empty images      : {m['n_empty_total']}  "
        f"correct={m['n_empty_correct']}  TNR={m['tnr']:.4f}\n"
    )


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── resolve dataset / split from --split argument ─────────────────────────
    parts = args.split.strip("/").split("/", 1)
    if len(parts) != 2:
        print(f"[Error] --split must be '<dataset>/<split_prefix>', got: {args.split!r}")
        sys.exit(1)
    dataset, split_prefix = parts

    dataset_root = os.path.join(args.datasets_root, dataset)
    args.output_dir = os.path.join(args.outputs_root, dataset, split_prefix)
    os.makedirs(args.output_dir, exist_ok=True)

    pore_csv = os.path.join(dataset_root, f"{split_prefix}_class2.csv")
    incl_csv = os.path.join(dataset_root, f"{split_prefix}_class3.csv")

    if not os.path.isfile(pore_csv):
        print(f"[Error] Pore CSV not found: {pore_csv}")
        sys.exit(1)

    pore_df = pd.read_csv(pore_csv)

    eval_incl = False
    incl_df   = None
    if args.pore_only:
        print(f"[INFO] --pore_only set — skipping inclusion evaluation.")
    elif not os.path.isfile(incl_csv):
        print(f"[INFO] {split_prefix}_class3.csv not found — skipping inclusion evaluation.")
    else:
        incl_df   = pd.read_csv(incl_csv)
        eval_incl = True

    # ── build models ──────────────────────────────────────────────────────────
    pore_device, incl_device = resolve_devices(args.pore_device, args.incl_device)
    print("[INFO] Building pore head …")
    pore_model = DamSAMHead(
        args.checkpoint_name, args.lora_r, args.lora_alpha, args.expert_num, args.image_size
    ).to(pore_device)
    print("[INFO] Building incl head …")
    incl_model = DamSAMHead(
        args.checkpoint_name, args.lora_r, args.lora_alpha, args.expert_num, args.image_size
    ).to(incl_device)
    load_joint_checkpoint(args.checkpoint, pore_model, incl_model, map_location="cpu")
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")
    print(f"[INFO] Dataset: {dataset}  Split: {split_prefix}  Eval incl: {eval_incl}")
    print(f"[INFO] Output dir: {args.output_dir}\n")

    # ── run evaluation ────────────────────────────────────────────────────────
    pore_m, incl_m = evaluate(
        pore_model, pore_device,
        incl_model, incl_device,
        pore_df, incl_df,
        dataset_root, eval_incl, args,
    )

    # ── print + save metrics ──────────────────────────────────────────────────
    pore_section = _section_str("Pore / class_2", f"{split_prefix}_class2", pore_m)
    print(pore_section)

    clahe_str = (f"clip={args.clahe_clip_limit} grid={args.clahe_grid_size}"
                 if args.apply_clahe else "off")
    header = (
        f"\n=== DAM-SAM Evaluation ===\n"
        f"checkpoint    : {args.checkpoint}\n"
        f"dataset       : {dataset}\n"
        f"split         : {split_prefix}\n"
        f"threshold     : pore={args.threshold_pore}  incl={args.threshold_incl}\n"
        f"morph         : pore={args.morph_size_pore}  incl={args.morph_size_incl}\n"
        f"blob_filter   : pore=[{args.min_blob_pore}, "
        f"{'∞' if args.max_blob_pore == 0 else args.max_blob_pore}]  "
        f"incl=[{args.min_blob_incl}, "
        f"{'∞' if args.max_blob_incl == 0 else args.max_blob_incl}]\n"
        f"clahe         : {clahe_str}\n"
        f"gamma         : {args.gamma}\n"
        f"tta           : {args.tta}\n"
    )
    summary = header + pore_section

    if eval_incl:
        incl_section = _section_str("Inclusion / class_3", f"{split_prefix}_class3", incl_m)
        print(incl_section)
        summary += incl_section

    metrics_path = os.path.join(args.output_dir, "inf_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(summary + "\n")
    print(f"Metrics saved to : {metrics_path}")
    if args.save_images:
        print(f"Images saved to  : {args.output_dir}/")


if __name__ == "__main__":
    main()


# python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split gan-generated/test1 --apply_clahe --clahe_clip_limit 0.03 --clahe_grid_size 8   --gamma 0.7 --threshold_pore 0.25 --threshold_incl 0.30  --tta  --morph_size_pore 0 --morph_size_incl 1   --min_blob_pore 2 --max_blob_pore 300   --min_blob_incl 2 --max_blob_incl 0
# python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split gan-generated/test2 --threshold_pore 0.35 --threshold_incl 0.60 --morph_size_pore 1 --morph_size_incl 7 --min_blob_pore 2 --max_blob_pore 0 --min_blob_incl 20 --max_blob_incl 0

# python3 scripts/evaluate.py  --checkpoint checkpoints/dam_sam/best.pt  --split gan-generated/test4 --pore_only
# python3 scripts/evaluate.py  --checkpoint checkpoints/dam_sam/best.pt --split gan-generated/test5 --pore_only --threshold_pore 0.60 --morph_size_pore 1 --min_blob_pore 12
# python3 scripts/evaluate.py  --checkpoint checkpoints/dam_sam/best.pt --split gan-generated/test6 --threshold_pore 0.15 --threshold_incl 0.75 --morph_size_pore 0 --morph_size_incl 1 --min_blob_pore 2 --max_blob_pore 200 --min_blob_incl 20


########### nist
# python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test2 --threshold_pore 0.90 --morph_size_pore 2 --erode_pore 1 --min_blob_pore 3 
# python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test3 
# python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test4 --threshold_pore 0.90 --morph_size_pore 2 --erode_pore 1 --min_blob_pore 3
# python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test5 
# python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test6 --threshold_pore 0.50 --morph_size_pore 0 --invert_pore --valid_region_mode nonzero