#!/usr/bin/env python3
"""
PTQ4SAM post-training quantization of a DAM-SAM joint checkpoint.

Implements the algorithm of "PTQ4SAM: Post-Training Quantization for Segment Anything"
(arXiv:2405.03144) -- Bimodal Integration + Adaptive Granularity Quantization -- against
DAM-SAM's HuggingFace-style SAM, then evaluates the quantized model with exactly the same
metric code as scripts/evaluate.py so FP and quantized numbers are directly comparable.

This is the *statistic-based* variant (PTQ4SAM-S in the paper): out-of-the-box calibration
with no block reconstruction. See --help for bit-width flags.

Example
  python3 scripts/quantize_ptq4sam.py --checkpoint checkpoints/dam_sam/best.pt \
      --split gan-generated/test4 --pore_only --w_bit 8 --a_bit 8
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.evaluate as ev                      # noqa: E402
from dam_sam import DamSAMHead                     # noqa: E402
from dam_sam.ptq4sam.calibrate import calibrate    # noqa: E402
from dam_sam.ptq4sam.quant_sam import PTQ4SAMQuantizer   # noqa: E402
from dam_sam.transforms import SAM_IMAGE_SIZE      # noqa: E402
from dam_sam.utils import load_joint_checkpoint, resolve_devices  # noqa: E402
from PIL import Image                              # noqa: E402


def build_cali_data(dataset_root, csv_name, n, args):
    """n calibration images, taken deterministically from the training split."""
    df = pd.read_csv(os.path.join(dataset_root, csv_name))
    step = max(1, len(df) // n)
    rows = df.iloc[::step].head(n)
    out = []
    for _, r in rows.iterrows():
        img = Image.open(os.path.join(dataset_root, r["image"]))
        out.append(ev.preprocess(img, args))       # identical preprocessing to evaluation
    return out


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--w_bit", type=int, default=8)
    ap.add_argument("--a_bit", type=int, default=8)
    ap.add_argument("--calib_size", type=int, default=32)
    ap.add_argument("--calib_csv", type=str, default="train_class2.csv")
    ap.add_argument("--no_big", action="store_true", help="ablate Bimodal Integration")
    ap.add_argument("--no_agq", action="store_true", help="ablate Adaptive Granularity Quantization")
    ap.add_argument("--fp", action="store_true", help="skip quantization (full-precision baseline)")
    ap.add_argument("--save_quant", type=str, default=None, help="path to save the quantized state dict")
    qargs, rest = ap.parse_known_args()

    sys.argv = [sys.argv[0]] + rest
    args = ev.parse_args()

    dataset, split_prefix = args.split.strip("/").split("/", 1)
    dataset_root = os.path.join(args.datasets_root, dataset)
    args.output_dir = os.path.join(args.outputs_root, dataset, split_prefix)
    os.makedirs(args.output_dir, exist_ok=True)

    pore_csv = os.path.join(dataset_root, f"{split_prefix}_class2.csv")
    incl_csv = os.path.join(dataset_root, f"{split_prefix}_class3.csv")
    pore_df = pd.read_csv(pore_csv)
    eval_incl = (not args.pore_only) and os.path.isfile(incl_csv)
    incl_df = pd.read_csv(incl_csv) if eval_incl else None

    pore_device, incl_device = resolve_devices(args.pore_device, args.incl_device)
    print("[INFO] Building heads …")
    pore_model = DamSAMHead(args.checkpoint_name, args.lora_r, args.lora_alpha,
                            args.expert_num, SAM_IMAGE_SIZE).to(pore_device)
    incl_model = DamSAMHead(args.checkpoint_name, args.lora_r, args.lora_alpha,
                            args.expert_num, SAM_IMAGE_SIZE).to(incl_device)
    load_joint_checkpoint(args.checkpoint, pore_model, incl_model, map_location="cpu")
    pore_model.eval(); incl_model.eval()

    tag = "FP32" if qargs.fp else f"W{qargs.w_bit}A{qargs.a_bit}"
    if not qargs.fp:
        cali = build_cali_data(dataset_root, qargs.calib_csv, qargs.calib_size, args)
        print(f"[INFO] calibration set: {len(cali)} images from {qargs.calib_csv}")
        for label, model in (("pore", pore_model), ("incl", incl_model)):
            if label == "incl" and not eval_incl:
                print("[INFO] skipping incl head (not evaluated)")
                continue
            print(f"\n[INFO] ── quantizing {label} head  ({tag}) ──")
            t0 = time.time()
            q = PTQ4SAMQuantizer(model.sam, w_bit=qargs.w_bit, a_bit=qargs.a_bit,
                                 use_big=not qargs.no_big, use_agq=not qargs.no_agq)
            calibrate(q, cali)
            print(f"[INFO] {label} calibration took {time.time() - t0:.1f}s")
            if label == "pore":
                pore_q = q
        if qargs.save_quant:
            torch.save({"pore": pore_model.state_dict(), "incl": incl_model.state_dict(),
                        "w_bit": qargs.w_bit, "a_bit": qargs.a_bit}, qargs.save_quant)
            print(f"[INFO] saved quantized weights -> {qargs.save_quant}")

    print(f"\n[INFO] Evaluating {tag} on {args.split} …")
    pore_m, incl_m = ev.evaluate(pore_model, pore_device, incl_model, incl_device,
                                 pore_df, incl_df, dataset_root, eval_incl, args)
    print(ev._section_str(f"Pore / class_2 [{tag}]", f"{split_prefix}_class2", pore_m))
    if eval_incl:
        print(ev._section_str(f"Inclusion / class_3 [{tag}]", f"{split_prefix}_class3", incl_m))


if __name__ == "__main__":
    main()
