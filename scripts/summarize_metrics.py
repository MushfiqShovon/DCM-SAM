"""
scripts/summarize_metrics.py
────────────────────────────────────────────────────────────────────────────────
Summarise DAM-SAM evaluation results across all datasets and test splits.

Expected directory layout:
    <outputs_root>/
        <dataset>/          e.g. gan-generated/, nist/
            <split>/        e.g. test1/, test2/, …
                inf_metrics.txt

Each inf_metrics.txt is produced by scripts/evaluate.py and contains one or two
section blocks (─── Pore / class_2  and optionally ─── Inclusion / class_3).

Usage:
    python3 scripts/summarize_metrics.py outputs/dam_sam
    python3 scripts/summarize_metrics.py outputs/dam_sam --datasets gan-generated nist
"""

import argparse
import glob
import os
import re
import sys

import numpy as np


# ─────────────────────────── parsing ──────────────────────────────────────────

def _parse_section(block: str):
    """
    Extract (iou, dice, f1, iou_pos, tnr) from one ─── section text block.
    Returns None if IoU line is absent (section not found / malformed).
    """
    m_iou = re.search(r"Mean IoU\s*:\s*([0-9.]+)", block)
    if not m_iou:
        return None
    iou = float(m_iou.group(1))

    m_dice = re.search(r"Mean Dice\s*:\s*([0-9.]+)", block)
    dice = float(m_dice.group(1)) if m_dice else 0.0

    m_f1 = re.search(r"Mean Tol-F1 \(5px\)\s*:\s*([0-9.]+)", block)
    f1 = float(m_f1.group(1)) if m_f1 else 0.0

    m_pos = re.search(r"IoU\(pos\)\s*:\s*([0-9.]+)", block)
    iou_pos = float(m_pos.group(1)) if m_pos else iou

    m_tnr = re.search(r"TNR\s*=\s*([0-9.]+)", block)
    tnr = float(m_tnr.group(1)) if m_tnr else None

    return iou, dice, f1, iou_pos, tnr


def parse_inf_metrics(txt_path):
    """
    Parse one inf_metrics.txt.
    Returns (pore_metrics, incl_metrics) where each is a 5-tuple or None.
    """
    if not os.path.isfile(txt_path):
        return None, None
    with open(txt_path) as fh:
        text = fh.read()

    # Each section runs from its ─── header to the next ─── header or EOF.
    pore_blocks = re.findall(r"─── Pore[^\n]*\n(.*?)(?=\n─── |\Z)", text, re.DOTALL)
    incl_blocks = re.findall(r"─── Inclusion[^\n]*\n(.*?)(?=\n─── |\Z)", text, re.DOTALL)

    pore = _parse_section(pore_blocks[-1]) if pore_blocks else None
    incl = _parse_section(incl_blocks[-1]) if incl_blocks else None
    return pore, incl


# ─────────────────────────── collection ───────────────────────────────────────

def collect_dataset(root, dataset):
    """
    Walk <root>/<dataset>/*/inf_metrics.txt and return:
        pore_rows : [(split, iou, dice, f1, iou_pos, tnr), …]
        incl_rows : [(split, iou, dice, f1, iou_pos, tnr), …]
    """
    pattern = os.path.join(root, dataset, "*", "inf_metrics.txt")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        return [], []

    pore_rows, incl_rows = [], []
    for path in paths:
        split = os.path.basename(os.path.dirname(path))
        pore, incl = parse_inf_metrics(path)
        if pore is None:
            print(f"  [warn] could not parse pore section: {path}")
        else:
            pore_rows.append((split, *pore))
        if incl is not None:
            incl_rows.append((split, *incl))
    return pore_rows, incl_rows


# ─────────────────────────── display ──────────────────────────────────────────

_COL_HDR = "  {:<10}  {:>7}  {:>7}  {:>8}  {:>9}  {:>7}"
_COL_ROW = "  {:<10}  {:>7.4f}  {:>7.4f}  {:>8.4f}  {:>9.4f}  {:>7}"
_SEP_LEN = 60


def _tnr_str(tnr):
    return f"{tnr:.4f}" if tnr is not None else "  —   "


def print_table(rows, label):
    if not rows:
        print(f"  No {label} results found.\n")
        return

    print(_COL_HDR.format("Split", "IoU", "Dice", "F1(5px)", "IoU(pos)", "TNR"))
    print("  " + "─" * _SEP_LEN)

    ious, dices, f1s, ious_pos = [], [], [], []
    for name, iou, dice, f1, iou_pos, tnr in rows:
        print(_COL_ROW.format(name, iou, dice, f1, iou_pos, _tnr_str(tnr)))
        ious.append(iou); dices.append(dice); f1s.append(f1); ious_pos.append(iou_pos)

    print("  " + "─" * _SEP_LEN)
    print(_COL_ROW.format("MEAN",
                          np.mean(ious), np.mean(dices),
                          np.mean(f1s),  np.mean(ious_pos), ""))
    print()


def print_dataset_block(root, dataset):
    pore_rows, incl_rows = collect_dataset(root, dataset)
    print(f"\n{'═' * (_SEP_LEN + 4)}")
    print(f"  {dataset.upper()}")
    print(f"{'═' * (_SEP_LEN + 4)}\n")

    print("─── Pore / class_2 " + "─" * (_SEP_LEN - 15))
    print_table(pore_rows, "Pore (class_2)")

    if incl_rows:
        pore_splits = {r[0] for r in pore_rows}
        incl_splits = {r[0] for r in incl_rows}
        skipped = sorted(pore_splits - incl_splits)
        print("─── Inclusion / class_3 " + "─" * (_SEP_LEN - 20))
        if skipped:
            print(f"  (not evaluated for: {', '.join(skipped)})\n")
        print_table(incl_rows, "Inclusion (class_3)")
    else:
        print("─── Inclusion / class_3 " + "─" * (_SEP_LEN - 20))
        print("  No inclusion labels for this dataset.\n")


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Summarise DAM-SAM evaluation results across all datasets and splits."
    )
    parser.add_argument(
        "outputs_root",
        help="Root output directory, e.g. 'outputs/dam_sam'. "
             "Expected layout: <outputs_root>/<dataset>/<split>/inf_metrics.txt",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to include (default: all subdirectories found under outputs_root).",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.outputs_root)
    if not os.path.isdir(root):
        print(f"Error: directory not found: {root}")
        sys.exit(1)

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )

    if not datasets:
        print(f"No dataset subdirectories found under: {root}")
        sys.exit(1)

    algo = os.path.basename(root)
    print(f"\nAlgorithm : {algo}")
    print(f"Root      : {root}")

    for dataset in datasets:
        print_dataset_block(root, dataset)

    print()


if __name__ == "__main__":
    main()


# Usage examples:
#   python3 scripts/summarize_metrics.py outputs/dam_sam
#   python3 scripts/summarize_metrics.py outputs/dam_sam --datasets gan-generated
#   python3 scripts/summarize_metrics.py outputs/dam_sam --datasets nist
