#!/usr/bin/env bash
# MoE expert-count ablation — inference for ALL 8 trained configurations
# (p1_i1, p1_i8, p2_i8, p4_i8, p8_i1, p8_i2, p8_i4, p8_i8) on the gan-generated
# test splits, in one command.
#
# Unlike run.sh, no per-split flags (--threshold_pore, --apply_clahe, --tta, …)
# are passed here: evaluate_moe.py has those settings HARDCODED per split
# (SPLIT_CONFIGS, copied verbatim from run.sh) and applies them identically to
# every run — that's what makes the ablation fair. Passing them on the command
# line is not supported and would error.
#
# evaluate_moe.py auto-discovers every subfolder of --checkpoints_root that
# contains a best.pt, reads its train_config.json for the per-head expert
# counts, and builds each head accordingly. Results are saved separately per
# configuration:
#
#   outputs/dam_sam_moe/<run_tag>/gan-generated/<split>/inf_metrics.txt
#   outputs/dam_sam_moe/ablation_summary.txt        (sweep tables, all runs)
#
# Per-run detail table afterwards:
#   python3 scripts/summarize_metrics.py outputs/dam_sam_moe/p1_i8

set -e

GAN_SPLITS="gan-generated/test1 gan-generated/test2 gan-generated/test4 gan-generated/test5 gan-generated/test6"

python3 scripts/evaluate_moe.py \
    --checkpoints_root checkpoints/dam_sam_moe \
    --outputs_root outputs/dam_sam_moe \
    --splits $GAN_SPLITS

# ── optional variants ─────────────────────────────────────────────────────────

# Single configuration only (e.g. re-run just p1_i1):
# python3 scripts/evaluate_moe.py --checkpoints_root checkpoints/dam_sam_moe \
#     --outputs_root outputs/dam_sam_moe --runs p1_i1 --splits $GAN_SPLITS

# Ranking-robustness check: default inference settings everywhere
# (writes to outputs/dam_sam_moe_vanilla/):
# python3 scripts/evaluate_moe.py --checkpoints_root checkpoints/dam_sam_moe \
#     --outputs_root outputs/dam_sam_moe --splits $GAN_SPLITS --vanilla

# Also save per-image panel figures (heavy — 8 runs x 5 splits):
# add --save_images to the main command above.
