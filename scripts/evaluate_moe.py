"""
scripts/evaluation_moe.py
────────────────────────────────────────────────────────────────────────────────
Evaluate every MoE-expert-count ablation run trained by scripts/train_moe.py.

For each run directory under --checkpoints_root (default checkpoints/dam_sam_moe)
containing a best.pt:

  1. Reads the run's train_config.json for expert_num_pore / expert_num_incl
     (falls back to parsing the run_tag "p{P}_i{I}") and builds each head with
     its own expert count — a head built with the wrong count cannot load the
     checkpoint, since the expert modules themselves differ.
  2. Evaluates on the FIXED per-split inference settings copied verbatim from
     scripts/run.sh (the settings calibrated once on the reference model).
     Nothing is re-tuned per run — differences in the tables come from the
     model, not the post-processing.
  3. Writes <outputs_root>/<run_tag>/<dataset>/<split>/inf_metrics.txt in the
     same format as scripts/evaluate.py, so scripts/summarize_metrics.py works
     on any single run root: python3 scripts/summarize_metrics.py outputs/dam_sam_moe/p1_i8

After all runs, prints (and saves to <outputs_root>/ablation_summary.txt) the
two sweep tables the ablation is for:
  - pore IoU vs. expert_num_pore   (runs with expert_num_incl = 8)
  - incl IoU vs. expert_num_incl   (runs with expert_num_pore = 8)
plus the p1_i1 corner run for the interaction check.

--vanilla re-runs everything with default inference settings (threshold 0.5,
no CLAHE/gamma/TTA/blob/erode), keeping only the structural per-split flags
(pore_only, invert_pore, valid_region_mode) — use it to confirm the expert-count
ranking is not an artifact of post-processing calibrated on the reference model.
Vanilla outputs go to <outputs_root>_vanilla/.

Notes
  - Panel images are OFF by default here (unlike evaluate.py): 8 runs x 10
    splits of per-image panels is a lot of disk. Pass --save_images to enable.
  - nist/test6 uses --invert_pore, which assumes the model segments solid
    material on that highly porous split. That is a behavior of the reference
    model — spot-check one prediction per run before trusting that row, or
    drop it from the paper table (it is flagged [invert] in the summary).

Usage
  python3 scripts/evaluation_moe.py                       # all runs, all splits
  python3 scripts/evaluation_moe.py --runs p1_i8 p8_i1    # subset of runs
  python3 scripts/evaluation_moe.py --splits nist/test2   # subset of splits
  python3 scripts/evaluation_moe.py --vanilla             # ranking-robustness check
"""

import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dam_sam import DamSAMHead
from dam_sam.transforms import SAM_IMAGE_SIZE, get_valid_region
from dam_sam.utils import load_joint_checkpoint, resolve_devices, tolerance_f1_masks

# Reuse the exact helpers evaluate.py uses so both scripts score identically.
from evaluate import _dice, _iou, _section_str, filter_components, predict_mask, preprocess, save_panel

_RUN_TAG_RE = re.compile(r"^p(\d+)_i(\d+)$")


# ─────────────────────────── fixed per-split settings ─────────────────────────
# Copied verbatim from scripts/run.sh — calibrated once on the reference model
# (checkpoints/dam_sam/best.pt) and held fixed for every ablation run.

_DEFAULTS = dict(
    threshold_pore=0.5, threshold_incl=0.5,
    morph_size_pore=1, morph_size_incl=1,
    apply_clahe=False, clahe_clip_limit=0.03, clahe_grid_size=8,
    gamma=1.0, tta=False,
    min_blob_pore=0, max_blob_pore=0, min_blob_incl=0, max_blob_incl=0,
    erode_pore=0, erode_incl=0,
    invert_pore=False, valid_region_mode="otsu", use_valid_region=True,
    pore_only=False,
)

# Structural flags survive --vanilla: they encode properties of the data/split
# (missing labels, highly porous specimen), not tuning of the reference model.
_STRUCTURAL_KEYS = {"pore_only", "invert_pore", "valid_region_mode"}

SPLIT_CONFIGS = [
    ("gan-generated/test1", dict(
        apply_clahe=True, clahe_clip_limit=0.03, clahe_grid_size=8, gamma=0.7,
        threshold_pore=0.25, threshold_incl=0.30, tta=True,
        morph_size_pore=0, morph_size_incl=1,
        min_blob_pore=2, max_blob_pore=300, min_blob_incl=2,
    )),
    ("gan-generated/test2", dict(
        threshold_pore=0.35, threshold_incl=0.60,
        morph_size_pore=1, morph_size_incl=7,
        min_blob_pore=2, min_blob_incl=20,
    )),
    ("gan-generated/test4", dict(pore_only=True)),
    ("gan-generated/test5", dict(
        pore_only=True, threshold_pore=0.60, morph_size_pore=1, min_blob_pore=12,
    )),
    ("gan-generated/test6", dict(
        threshold_pore=0.15, threshold_incl=0.75,
        morph_size_pore=0, morph_size_incl=1,
        min_blob_pore=2, max_blob_pore=200, min_blob_incl=20,
    )),
    ("nist/test2", dict(
        threshold_pore=0.90, morph_size_pore=2, erode_pore=1, min_blob_pore=3,
    )),
    ("nist/test3", dict()),
    ("nist/test4", dict(
        threshold_pore=0.90, morph_size_pore=2, erode_pore=1, min_blob_pore=3,
    )),
    ("nist/test5", dict()),
    ("nist/test6", dict(
        threshold_pore=0.50, morph_size_pore=0,
        invert_pore=True, valid_region_mode="nonzero",
    )),
]


def build_split_cfg(overrides, vanilla, common):
    if vanilla:
        overrides = {k: v for k, v in overrides.items() if k in _STRUCTURAL_KEYS}
    cfg = SimpleNamespace(**{**_DEFAULTS, **overrides})
    cfg.tol_f1_tolerance = common.tol_f1_tolerance
    cfg.image_size = getattr(common, "image_size", SAM_IMAGE_SIZE)
    return cfg


# ─────────────────────────── CLI ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate all MoE-ablation checkpoints with fixed per-split inference settings."
    )
    p.add_argument("--checkpoints_root", type=str, default="checkpoints/dam_sam_vitb_moe",
                   help="Directory of run subfolders, each containing best.pt + train_config.json.")
    p.add_argument("--outputs_root", type=str, default="outputs/dam_sam_vitb_moe",
                   help="Results go to <outputs_root>/<run_tag>/<dataset>/<split>/. "
                        "With --vanilla, '_vanilla' is appended to this root.")
    p.add_argument("--datasets_root", type=str, default="datasets")
    p.add_argument("--runs", type=str, nargs="+", default=None,
                   help="Run tags to evaluate (e.g. p1_i8 p8_i1); default: every subfolder with a best.pt.")
    p.add_argument("--splits", type=str, nargs="+", default=None,
                   help="Subset of splits (e.g. gan-generated/test1 nist/test2); default: all 10.")
    p.add_argument("--vanilla", action="store_true", default=False,
                   help="Default inference settings everywhere (only structural per-split flags kept) — "
                        "ranking-robustness check against reference-model-calibrated post-processing.")
    p.add_argument("--checkpoint_name", type=str, default="facebook/sam-vit-base")
    p.add_argument("--image_size", type=int, default=SAM_IMAGE_SIZE,
                   help="Square input resolution; pos_embed is resampled from 1024.")
    p.add_argument("--lora_r", type=int, default=2)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--pore_device", type=str, default=None)
    p.add_argument("--incl_device", type=str, default=None)
    p.add_argument("--tol_f1_tolerance", type=int, default=5)
    p.add_argument("--save_images", action="store_true", default=False,
                   help="Save per-image panels (off by default — heavy across 8 runs x 10 splits).")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


# ─────────────────────────── run discovery ────────────────────────────────────

def discover_runs(args):
    """Returns [(run_tag, ckpt_path, expert_num_pore, expert_num_incl), ...] sorted by tag."""
    if args.runs:
        tags = args.runs
    else:
        tags = sorted(
            d for d in os.listdir(args.checkpoints_root)
            if os.path.isfile(os.path.join(args.checkpoints_root, d, "best.pt"))
        )
    runs = []
    for tag in tags:
        run_dir = os.path.join(args.checkpoints_root, tag)
        ckpt = os.path.join(run_dir, "best.pt")
        if not os.path.isfile(ckpt):
            print(f"[WARN] {ckpt} not found — skipping run {tag}")
            continue
        cfg_path = os.path.join(run_dir, "train_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as fh:
                tc = json.load(fh)
            ep, ei = tc["expert_num_pore"], tc["expert_num_incl"]
        else:
            m = _RUN_TAG_RE.match(tag)
            if not m:
                print(f"[WARN] {tag}: no train_config.json and tag is not p{{P}}_i{{I}} — skipping")
                continue
            ep, ei = int(m.group(1)), int(m.group(2))
            print(f"[WARN] {tag}: no train_config.json, expert counts parsed from tag (pore={ep}, incl={ei})")
        runs.append((tag, ckpt, ep, ei))
    # Numeric order (p1, p2, p4, p8) rather than lexicographic (p1, p2, p4, p8 happens to
    # agree here, but keep it robust to future counts like 16).
    runs.sort(key=lambda r: (r[2], r[3]))
    return runs


# ─────────────────────────── extra mask ops (not in evaluate.py) ──────────────

def valid_region_nonzero(gray, threshold=5):
    """Specimen mask for particle-dominated images where Otsu latches onto bright
    clusters instead of the specimen boundary (nist/test6)."""
    mask = gray > threshold
    mask = ndi.binary_closing(mask, structure=np.ones((15, 15)))
    return ndi.binary_fill_holes(mask)


def postprocess_binary(binary, erode, min_blob, max_blob):
    if erode > 0:
        binary = ndi.binary_erosion(binary, iterations=erode).astype(np.uint8)
    return filter_components(binary, min_blob, max_blob)


# ─────────────────────────── per-split evaluation ─────────────────────────────

def evaluate_split(pore_model, pore_device, incl_model, incl_device,
                   dataset_root, split_prefix, cfg, out_dir, common):
    """Same scoring loop as evaluate.py, extended with erosion, invert_pore, and
    the nonzero valid-region mode that the fixed nist settings need.
    Returns (pore_m, incl_m_or_None) metric dicts."""
    pore_csv = os.path.join(dataset_root, f"{split_prefix}_class2.csv")
    incl_csv = os.path.join(dataset_root, f"{split_prefix}_class3.csv")
    if not os.path.isfile(pore_csv):
        print(f"[Error] Pore CSV not found: {pore_csv}")
        return None, None
    pore_df = pd.read_csv(pore_csv)

    eval_incl = (not cfg.pore_only) and os.path.isfile(incl_csv)
    incl_df = pd.read_csv(incl_csv) if eval_incl else None
    incl_iter = iter(incl_df.iterrows()) if eval_incl else None

    pore_ious, pore_dices, pore_precs, pore_recs, pore_f1s, pore_ious_pos = [], [], [], [], [], []
    p_empty_total = p_empty_correct = 0
    incl_ious, incl_dices, incl_precs, incl_recs, incl_f1s, incl_ious_pos = [], [], [], [], [], []
    i_empty_total = i_empty_correct = 0

    for i, (_, pore_row) in enumerate(
        tqdm(pore_df.iterrows(), total=len(pore_df), desc=f"  {split_prefix}", unit="img")
    ):
        img_path = os.path.join(dataset_root, pore_row["image"])
        img_pil = Image.open(img_path)
        gray = np.array(img_pil.convert("L"))
        orig_h, orig_w = gray.shape[:2]
        pv = preprocess(img_pil, cfg)

        valid = None
        if cfg.use_valid_region:
            valid = (valid_region_nonzero(gray) if cfg.valid_region_mode == "nonzero"
                     else get_valid_region(gray))

        # ── pore head ────────────────────────────────────────────────────────
        pore_binary = predict_mask(
            pore_model, pore_device, pv,
            cfg.morph_size_pore, cfg.threshold_pore, orig_h, orig_w, tta=cfg.tta,
        )
        pore_binary = postprocess_binary(pore_binary, cfg.erode_pore, cfg.min_blob_pore, cfg.max_blob_pore)
        if cfg.invert_pore:
            pore_binary = np.where(valid, 1 - pore_binary, 0).astype(np.uint8)
        pore_gt_raw = np.array(Image.open(os.path.join(dataset_root, pore_row["label"])).convert("L")) > 0

        if cfg.use_valid_region:
            pore_pred = np.where(valid, pore_binary, 0).astype(bool)
            pore_gt = np.where(valid, pore_gt_raw, 0).astype(bool)
        else:
            pore_pred = pore_binary.astype(bool)
            pore_gt = pore_gt_raw

        p_iou, p_dice = _iou(pore_pred, pore_gt), _dice(pore_pred, pore_gt)
        p_prec, p_rec, p_f1 = tolerance_f1_masks(pore_pred, pore_gt, tolerance=cfg.tol_f1_tolerance)
        pore_ious.append(p_iou); pore_dices.append(p_dice)
        pore_precs.append(p_prec); pore_recs.append(p_rec); pore_f1s.append(p_f1)
        if pore_gt.any():
            pore_ious_pos.append(p_iou)
        else:
            p_empty_total += 1
            if not pore_pred.any():
                p_empty_correct += 1

        # ── inclusion head ───────────────────────────────────────────────────
        incl_gt_raw = incl_binary = None
        i_iou = i_dice = i_f1 = None
        if eval_incl:
            _, incl_row = next(incl_iter)
            incl_binary = predict_mask(
                incl_model, incl_device, pv,
                cfg.morph_size_incl, cfg.threshold_incl, orig_h, orig_w, tta=cfg.tta,
            )
            incl_binary = postprocess_binary(incl_binary, cfg.erode_incl, cfg.min_blob_incl, cfg.max_blob_incl)
            incl_gt_raw = np.array(Image.open(os.path.join(dataset_root, incl_row["label"])).convert("L")) > 0

            if cfg.use_valid_region:
                incl_pred = np.where(valid, incl_binary, 0).astype(bool)
                incl_gt = np.where(valid, incl_gt_raw, 0).astype(bool)
            else:
                incl_pred = incl_binary.astype(bool)
                incl_gt = incl_gt_raw

            i_iou, i_dice = _iou(incl_pred, incl_gt), _dice(incl_pred, incl_gt)
            i_prec, i_rec, i_f1 = tolerance_f1_masks(incl_pred, incl_gt, tolerance=cfg.tol_f1_tolerance)
            incl_ious.append(i_iou); incl_dices.append(i_dice)
            incl_precs.append(i_prec); incl_recs.append(i_rec); incl_f1s.append(i_f1)
            if incl_gt.any():
                incl_ious_pos.append(i_iou)
            else:
                i_empty_total += 1
                if not incl_pred.any():
                    i_empty_correct += 1

        if common.save_images:
            save_panel(
                gray, pore_gt_raw, pore_binary, incl_gt_raw, incl_binary,
                f"joint_{os.path.basename(img_path)}", out_dir,
                p_iou, p_dice, p_f1, i_iou, i_dice, i_f1,
            )
        if (i + 1) % common.log_every == 0:
            print(f"    [{i + 1}/{len(pore_df)}]  pore IoU={np.mean(pore_ious):.4f}"
                  + (f"  incl IoU={np.mean(incl_ious):.4f}" if eval_incl else ""))

    p_tnr = p_empty_correct / p_empty_total if p_empty_total > 0 else 1.0
    pore_m = dict(ious=pore_ious, dices=pore_dices, f1s=pore_f1s,
                  precs=pore_precs, recs=pore_recs, ious_pos=pore_ious_pos,
                  n_empty_total=p_empty_total, n_empty_correct=p_empty_correct, tnr=p_tnr)
    incl_m = None
    if eval_incl:
        i_tnr = i_empty_correct / i_empty_total if i_empty_total > 0 else 1.0
        incl_m = dict(ious=incl_ious, dices=incl_dices, f1s=incl_f1s,
                      precs=incl_precs, recs=incl_recs, ious_pos=incl_ious_pos,
                      n_empty_total=i_empty_total, n_empty_correct=i_empty_correct, tnr=i_tnr)
    return pore_m, incl_m


def write_metrics_file(out_dir, run_tag, ep, ei, ckpt, dataset, split_prefix, cfg, vanilla, pore_m, incl_m):
    clahe_str = (f"clip={cfg.clahe_clip_limit} grid={cfg.clahe_grid_size}" if cfg.apply_clahe else "off")
    header = (
        f"\n=== DAM-SAM MoE-Ablation Evaluation ===\n"
        f"run_tag       : {run_tag}  (expert_num_pore={ep}  expert_num_incl={ei})\n"
        f"checkpoint    : {ckpt}\n"
        f"dataset       : {dataset}\n"
        f"split         : {split_prefix}\n"
        f"settings      : {'VANILLA (defaults, structural flags only)' if vanilla else 'fixed per-split (run.sh)'}\n"
        f"threshold     : pore={cfg.threshold_pore}  incl={cfg.threshold_incl}\n"
        f"morph         : pore={cfg.morph_size_pore}  incl={cfg.morph_size_incl}\n"
        f"erode         : pore={cfg.erode_pore}  incl={cfg.erode_incl}\n"
        f"blob_filter   : pore=[{cfg.min_blob_pore}, {'∞' if cfg.max_blob_pore == 0 else cfg.max_blob_pore}]  "
        f"incl=[{cfg.min_blob_incl}, {'∞' if cfg.max_blob_incl == 0 else cfg.max_blob_incl}]\n"
        f"clahe         : {clahe_str}\n"
        f"gamma         : {cfg.gamma}\n"
        f"tta           : {cfg.tta}\n"
        f"invert_pore   : {cfg.invert_pore}\n"
        f"valid_region  : {cfg.valid_region_mode}\n"
    )
    summary = header + _section_str("Pore / class_2", f"{split_prefix}_class2", pore_m)
    if incl_m is not None:
        summary += _section_str("Inclusion / class_3", f"{split_prefix}_class3", incl_m)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "inf_metrics.txt")
    with open(path, "w") as f:
        f.write(summary + "\n")
    return path


# ─────────────────────────── cross-run summary ────────────────────────────────

def summarize(all_results, runs, outputs_root, vanilla):
    """all_results[tag] = list of (dataset, split, pore_m, incl_m_or_None)."""
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    def agg(tag, dataset, head):
        vals = [np.mean((p if head == "pore" else i)["ious"])
                for d, s, p, i in all_results.get(tag, [])
                if d == dataset and (head == "pore" or i is not None)]
        return np.mean(vals) if vals else None

    def fmt(v):
        return f"{v:.4f}" if v is not None else "   —  "

    experts_of = {tag: (ep, ei) for tag, _, ep, ei in runs}

    emit()
    emit("═" * 72)
    emit(f"  MoE EXPERT-COUNT ABLATION SUMMARY"
         + ("  [VANILLA settings]" if vanilla else "  [fixed run.sh settings]"))
    emit("  Cell values: mean over that dataset's splits of per-split Mean IoU")
    emit("═" * 72)

    emit()
    emit("  Per-run overview")
    emit(f"  {'run':<8} {'experts(p,i)':<13} {'gan pore':>9} {'gan incl':>9} {'nist pore':>10}")
    emit("  " + "─" * 54)
    for tag, _, ep, ei in runs:
        emit(f"  {tag:<8} ({ep},{ei}){'':<7} {fmt(agg(tag, 'gan-generated', 'pore')):>9} "
             f"{fmt(agg(tag, 'gan-generated', 'incl')):>9} {fmt(agg(tag, 'nist', 'pore')):>10}")

    pore_sweep = sorted([t for t, (ep, ei) in experts_of.items() if ei == 8], key=lambda t: experts_of[t][0])
    incl_sweep = sorted([t for t, (ep, ei) in experts_of.items() if ep == 8], key=lambda t: experts_of[t][1])
    corner = [t for t, (ep, ei) in experts_of.items() if ep != 8 and ei != 8]

    if pore_sweep:
        emit()
        emit("  Pore sweep (expert_num_incl = 8) — read pore columns")
        emit(f"  {'pore experts':<14} {'gan pore IoU':>13} {'nist pore IoU':>14}")
        emit("  " + "─" * 43)
        for tag in pore_sweep:
            emit(f"  {experts_of[tag][0]:<14} {fmt(agg(tag, 'gan-generated', 'pore')):>13} "
                 f"{fmt(agg(tag, 'nist', 'pore')):>14}")

    if incl_sweep:
        emit()
        emit("  Inclusion sweep (expert_num_pore = 8) — read incl column")
        emit(f"  {'incl experts':<14} {'gan incl IoU':>13}")
        emit("  " + "─" * 29)
        for tag in incl_sweep:
            emit(f"  {experts_of[tag][1]:<14} {fmt(agg(tag, 'gan-generated', 'incl')):>13}")

    if corner:
        emit()
        emit(f"  Corner run(s) for interaction check: {', '.join(corner)} (see per-run overview)")

    emit()
    emit("  nist/test6 rows use invert_pore [invert] — verify the inverted-prediction")
    emit("  assumption holds per run before citing them (see module docstring).")
    emit()

    os.makedirs(outputs_root, exist_ok=True)
    summary_path = os.path.join(outputs_root, "ablation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary saved to: {summary_path}")


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    args = parse_args()
    outputs_root = args.outputs_root + ("_vanilla" if args.vanilla else "")

    split_configs = SPLIT_CONFIGS
    if args.splits:
        wanted = set(args.splits)
        split_configs = [(s, o) for s, o in SPLIT_CONFIGS if s in wanted]
        unknown = wanted - {s for s, _ in SPLIT_CONFIGS}
        if unknown:
            print(f"[Error] Unknown split(s): {sorted(unknown)}. Known: {[s for s, _ in SPLIT_CONFIGS]}")
            sys.exit(1)

    runs = discover_runs(args)
    if not runs:
        print(f"[Error] No runs with best.pt found under {args.checkpoints_root}")
        sys.exit(1)
    print(f"[INFO] Runs to evaluate: {[t for t, *_ in runs]}")
    print(f"[INFO] Splits: {[s for s, _ in split_configs]}")
    print(f"[INFO] Settings: {'VANILLA' if args.vanilla else 'fixed per-split (run.sh)'}  "
          f"Outputs: {outputs_root}/\n")

    pore_device, incl_device = resolve_devices(args.pore_device, args.incl_device)
    all_results = {}

    for tag, ckpt, ep, ei in runs:
        print(f"\n──── run {tag} (pore experts={ep}, incl experts={ei}) ────")
        pore_model = DamSAMHead(args.checkpoint_name, args.lora_r, args.lora_alpha, ep, args.image_size).to(pore_device)
        incl_model = DamSAMHead(args.checkpoint_name, args.lora_r, args.lora_alpha, ei, args.image_size).to(incl_device)
        load_joint_checkpoint(ckpt, pore_model, incl_model, map_location="cpu")
        pore_model.eval()
        incl_model.eval()
        print(f"[INFO] Loaded {ckpt}")

        results = []
        for split, overrides in split_configs:
            dataset, split_prefix = split.split("/", 1)
            cfg = build_split_cfg(overrides, args.vanilla, args)
            out_dir = os.path.join(outputs_root, tag, dataset, split_prefix)
            pore_m, incl_m = evaluate_split(
                pore_model, pore_device, incl_model, incl_device,
                os.path.join(args.datasets_root, dataset), split_prefix, cfg, out_dir, args,
            )
            if pore_m is None:
                continue
            path = write_metrics_file(out_dir, tag, ep, ei, ckpt, dataset, split_prefix,
                                      cfg, args.vanilla, pore_m, incl_m)
            print(f"  {split}: pore IoU={np.mean(pore_m['ious']):.4f}"
                  + (f"  incl IoU={np.mean(incl_m['ious']):.4f}" if incl_m else "")
                  + f"  → {path}")
            results.append((dataset, split_prefix, pore_m, incl_m))
        all_results[tag] = results

        # Free the ~2.6GB x2 of ViT-H before building the next run's heads.
        del pore_model, incl_model
        torch.cuda.empty_cache()

    summarize(all_results, runs, outputs_root, args.vanilla)


if __name__ == "__main__":
    main()

# python3 scripts/evaluation_moe.py
# python3 scripts/evaluation_moe.py --vanilla
# python3 scripts/summarize_metrics.py outputs/dam_sam_moe/p1_i8   # per-run detail
