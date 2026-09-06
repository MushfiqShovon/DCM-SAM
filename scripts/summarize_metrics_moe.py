"""
scripts/summarize_metrics_moe.py
────────────────────────────────────────────────────────────────────────────────
Summarise DAM-SAM MoE-expert-count ablation results across all runs, datasets,
and test splits.

Directory layout (one level deeper than scripts/summarize_metrics.py, which
this script builds on and imports from):

    <outputs_root>/
        <run_tag>/            e.g. p1_i8, p8_i1, p8_i8, p2_i4, ...
            <dataset>/        e.g. gan-generated/
                <split>/      e.g. test1/, test2/, ...
                    inf_metrics.txt

For each run_tag this reuses summarize_metrics.collect_dataset() /
summarize_metrics.print_table() unchanged (a run_tag directory has exactly the
layout summarize_metrics.py already expects, one directory down), so per-run
tables always match what `python3 scripts/summarize_metrics.py outputs/dam_sam_moe/<run_tag>`
would print standalone.

On top of that, this script adds the piece the ablation itself is for: a
cross-run comparison, one row per run_tag, showing mean Pore IoU and mean
Inclusion IoU aggregated over all evaluated splits of a dataset — sorted by
the run's (pore_experts, incl_experts) parsed from its tag, so the pore and
inclusion sweeps read top-to-bottom as increasing expert count.

Usage:
    python3 scripts/summarize_metrics_moe.py outputs/dam_sam_moe
    python3 scripts/summarize_metrics_moe.py outputs/dam_sam_moe --runs p1_i8 p8_i1 p8_i8
    python3 scripts/summarize_metrics_moe.py outputs/dam_sam_moe --datasets gan-generated
"""

import argparse
import os
import re
import sys

import numpy as np

# Reuses the exact parsing/printing logic scripts/summarize_metrics.py uses for a single
# run's <dataset>/<split>/inf_metrics.txt tree, so per-run tables here and the standalone
# `summarize_metrics.py outputs/dam_sam_moe/<run_tag>` output never drift apart.
from summarize_metrics import collect_dataset, print_table

_RUN_TAG_RE = re.compile(r"^p(\d+)_i(\d+)$")


def parse_run_tag(tag):
    """Returns (pore_experts, incl_experts) or (None, None) if the tag doesn't match
    the p{P}_i{I} convention used by scripts/train_moe.py."""
    m = _RUN_TAG_RE.match(tag)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def sort_key(tag):
    ep, ei = parse_run_tag(tag)
    if ep is None:
        return (1, tag)  # unparsed tags sort after, alphabetically
    return (0, ep, ei)


# ─────────────────────────── run discovery ────────────────────────────────────

def discover_run_tags(root):
    return sorted(
        (d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))),
        key=sort_key,
    )


def discover_datasets(root, run_tags):
    """Union of dataset subdirectories across all runs (a run may have been evaluated
    on fewer datasets than another, e.g. a partial/failed run)."""
    found = set()
    for tag in run_tags:
        run_dir = os.path.join(root, tag)
        found.update(
            d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))
        )
    return sorted(found)


# ─────────────────────────── per-run detail ────────────────────────────────────

def print_run_block(root, run_tag, datasets):
    ep, ei = parse_run_tag(run_tag)
    experts_str = f"(pore experts={ep}, incl experts={ei})" if ep is not None else "(unparsed tag)"

    print(f"\n{'#' * 76}")
    print(f"  RUN: {run_tag}  {experts_str}")
    print(f"{'#' * 76}")

    run_dir = os.path.join(root, run_tag)
    for dataset in datasets:
        pore_rows, incl_rows = collect_dataset(run_dir, dataset)
        if not pore_rows and not incl_rows:
            continue

        print(f"\n{'─' * 64}")
        print(f"  {dataset}")
        print(f"{'─' * 64}\n")

        print("  Pore / class_2")
        print_table(pore_rows, "Pore (class_2)")

        if incl_rows:
            print("  Inclusion / class_3")
            print_table(incl_rows, "Inclusion (class_3)")


# ─────────────────────────── cross-run sweep tables ───────────────────────────

def cross_run_means(root, run_tags, dataset):
    """{run_tag: (pore_mean_iou_or_None, incl_mean_iou_or_None)} averaged over that
    dataset's evaluated splits."""
    means = {}
    for tag in run_tags:
        pore_rows, incl_rows = collect_dataset(os.path.join(root, tag), dataset)
        pore_mean = np.mean([r[1] for r in pore_rows]) if pore_rows else None
        incl_mean = np.mean([r[1] for r in incl_rows]) if incl_rows else None
        means[tag] = (pore_mean, incl_mean)
    return means


def _fmt(v):
    return f"{v:.4f}" if v is not None else "   —  "


def print_cross_run_table(root, run_tags, dataset):
    means = cross_run_means(root, run_tags, dataset)

    print(f"\n{'═' * 76}")
    print(f"  CROSS-RUN SWEEP — {dataset}")
    print(f"  (mean IoU over all evaluated splits of this dataset)")
    print(f"{'═' * 76}\n")

    print(f"  {'run':<10} {'pore experts':>12} {'incl experts':>12} {'pore IoU':>10} {'incl IoU':>10}")
    print("  " + "─" * 58)
    for tag in run_tags:
        ep, ei = parse_run_tag(tag)
        p_mean, i_mean = means[tag]
        ep_str = str(ep) if ep is not None else "?"
        ei_str = str(ei) if ei is not None else "?"
        print(f"  {tag:<10} {ep_str:>12} {ei_str:>12} {_fmt(p_mean):>10} {_fmt(i_mean):>10}")

    # Named sweeps, read off the same means: pore sweep = incl experts held at 8,
    # incl sweep = pore experts held at 8 — matches the runs scripts/train_moe.py
    # was designed to produce.
    pore_sweep = sorted(
        (t for t in run_tags if parse_run_tag(t)[1] == 8), key=lambda t: parse_run_tag(t)[0]
    )
    incl_sweep = sorted(
        (t for t in run_tags if parse_run_tag(t)[0] == 8), key=lambda t: parse_run_tag(t)[1]
    )
    corner = [t for t in run_tags if all(parse_run_tag(t)) and parse_run_tag(t) != (8, 8)
              and t not in pore_sweep and t not in incl_sweep]

    if pore_sweep:
        print(f"\n  Pore sweep (incl experts = 8) — pore IoU vs. pore expert count")
        print(f"  {'pore experts':<14} {'pore IoU':>10}")
        print("  " + "─" * 25)
        for tag in pore_sweep:
            ep, _ = parse_run_tag(tag)
            print(f"  {ep:<14} {_fmt(means[tag][0]):>10}")

    if incl_sweep:
        print(f"\n  Inclusion sweep (pore experts = 8) — incl IoU vs. incl expert count")
        print(f"  {'incl experts':<14} {'incl IoU':>10}")
        print("  " + "─" * 25)
        for tag in incl_sweep:
            _, ei = parse_run_tag(tag)
            print(f"  {ei:<14} {_fmt(means[tag][1]):>10}")

    if corner:
        print(f"\n  Other run(s) (interaction / extra points): {', '.join(corner)}")
    print()


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Summarise DAM-SAM MoE expert-count ablation results across all runs, "
        "datasets, and splits."
    )
    parser.add_argument(
        "outputs_root",
        help="Root output directory, e.g. 'outputs/dam_sam_moe'. "
             "Expected layout: <outputs_root>/<run_tag>/<dataset>/<split>/inf_metrics.txt",
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Run tags to include (default: every subdirectory found under outputs_root).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to include (default: union of all datasets found across the selected runs).",
    )
    parser.add_argument(
        "--no_detail", action="store_true", default=False,
        help="Skip the per-run detail tables and print only the cross-run sweep tables.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.outputs_root)
    if not os.path.isdir(root):
        print(f"Error: directory not found: {root}")
        sys.exit(1)

    run_tags = args.runs if args.runs else discover_run_tags(root)
    if not run_tags:
        print(f"No run subdirectories found under: {root}")
        sys.exit(1)

    datasets = args.datasets if args.datasets else discover_datasets(root, run_tags)
    if not datasets:
        print(f"No dataset subdirectories found under any run in: {root}")
        sys.exit(1)

    print(f"\nAblation root : {root}")
    print(f"Runs          : {', '.join(run_tags)}")
    print(f"Datasets      : {', '.join(datasets)}")

    if not args.no_detail:
        for tag in run_tags:
            print_run_block(root, tag, datasets)

    for dataset in datasets:
        print_cross_run_table(root, run_tags, dataset)


if __name__ == "__main__":
    main()


# Usage examples:
#   python3 scripts/summarize_metrics_moe.py outputs/dam_sam_moe
#   python3 scripts/summarize_metrics_moe.py outputs/dam_sam_moe --runs p1_i8 p2_i8 p4_i8 p8_i8
#   python3 scripts/summarize_metrics_moe.py outputs/dam_sam_moe --datasets gan-generated
#   python3 scripts/summarize_metrics_moe.py outputs/dam_sam_moe --no_detail   # sweep tables only
