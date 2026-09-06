#!/usr/bin/env bash
# ViT-B MoE expert-count ablation -- inference for all 8 trained configurations
# (p1_i1, p1_i8, p2_i8, p4_i8, p8_i1, p8_i2, p8_i4, p8_i8) on the gan-generated
# test splits, in one command. Mirrors scripts/run_moe.sh exactly, pointed at
# the ViT-B checkpoint/output roots so it never touches the ViT-H results.
#
# Same rule as run_moe.sh applies: no per-split flags on the command line,
# evaluate_moe.py hardcodes them per split (SPLIT_CONFIGS) and applies them
# identically to every run so the ablation stays fair.
#
#   outputs/dam_sam_vitb_moe/<run_tag>/gan-generated/<split>/inf_metrics.txt
#   outputs/dam_sam_vitb_moe/ablation_summary.txt        (sweep tables, all runs)
#
# Per-run detail table afterwards:
#   python3 scripts/summarize_metrics.py outputs/dam_sam_vitb_moe/p1_i8

set -e

GAN_SPLITS="gan-generated/test1 gan-generated/test2 gan-generated/test4 gan-generated/test5 gan-generated/test6"

python3 scripts/evaluate_moe.py \
    --checkpoint_name facebook/sam-vit-base \
    --checkpoints_root checkpoints/dam_sam_vitb_moe \
    --outputs_root outputs/dam_sam_vitb_moe \
    --splits $GAN_SPLITS

# ── optional variants ─────────────────────────────────────────────────────────

# Single configuration only:
# python3 scripts/evaluate_moe.py --checkpoint_name facebook/sam-vit-base \
#     --checkpoints_root checkpoints/dam_sam_vitb_moe \
#     --outputs_root outputs/dam_sam_vitb_moe --runs p1_i1 --splits $GAN_SPLITS

# Ranking-robustness check: default inference settings everywhere
# (writes to outputs/dam_sam_vitb_moe_vanilla/):
# python3 scripts/evaluate_moe.py --checkpoint_name facebook/sam-vit-base \
#     --checkpoints_root checkpoints/dam_sam_vitb_moe \
#     --outputs_root outputs/dam_sam_vitb_moe --splits $GAN_SPLITS --vanilla

# Also save per-image panel figures:
# add --save_images to the main command above.
