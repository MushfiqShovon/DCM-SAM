# DAM-SAM

Reference implementation for *DAM-SAM: Defect-Conditioned Adaptive Mixture of LoRA Experts for
On-Device AM Defect Segmentation*.

One frozen Segment Anything backbone hosts a separate Conv-LoRA expert bank and mask decoder for
each defect class. The two heads are trained without prompts, in separate passes, and the whole
model runs at full 1024x1024 resolution on a Qualcomm Hexagon NPU with every operator on the
accelerator.

## Setup

The vendored SAM implementation in `dam_sam/modeling_sam_conv_lora.py` is copied from
transformers v4 and **will not run on transformers 5.x**: v5 changed `_tied_weights_keys` from a
list to a dict, and every run dies in `post_init()` with
`AttributeError: 'list' object has no attribute 'keys'`. Pin it:

```bash
conda create -p ./env python=3.11 -y
conda activate ./env
pip install -r requirements.txt
```

A CUDA build of PyTorch matching your GPU is required; the pinned version in
`requirements.txt` targets CUDA 13 (Blackwell). Adjust it for other hardware.

## Data layout

```
datasets/
  gan-generated/          CycleGAN-synthesised XCT slices (train, test1..test6)
    <split>_class2.csv    pore masks
    <split>_class3.csv    inclusion masks
  nist/                   real NIST AM XCT specimens, pore labels only
```

Each CSV lists image and mask paths. `test4` and `test5` carry pore labels only.

## Training

One head per defect class, each with its own expert count. `p2_i8` below is the configuration
reported in the paper: two experts for pores, eight for inclusions.

```bash
python3 scripts/train_moe.py \
    --checkpoint_name facebook/sam-vit-base \
    --expert_num_pore 2 --expert_num_incl 8 \
    --output_dir checkpoints/dam_sam_vitb_moe --run_tag p2_i8 \
    --epochs 30 --val_split test6
```

Only the Conv-LoRA parameters and the mask decoders are updated; the backbone and prompt encoder
stay frozen. Note that `--val_split test6` is also one of the reported test splits, so the
five-split means include the selection split.

## Evaluation

Every configuration is scored with a fixed per-split post-processing configuration that
`scripts/evaluate_moe.py` hardcodes in `SPLIT_CONFIGS`, applied identically to every model so the
comparison isolates the model rather than the threshold.

```bash
# all trained configurations, all synthetic splits, plus the ablation summary
bash scripts/run_moe_vitb.sh

# one configuration on one split
python3 scripts/evaluate.py \
    --checkpoint checkpoints/dam_sam_vitb_moe/p2_i8/best.pt \
    --checkpoint_name facebook/sam-vit-base \
    --split gan-generated/test1
```

IoU is averaged over images, and an image with no defect and no predicted pixel scores 1. The
per-split reports also print `IoU(pos)`, the average over defect-bearing images only, which is the
number to read for segmentation quality on sparse splits.

## On-device export

Two rewrites are needed before the model can be traced into a static graph for the NPU, both in
this repository and both numerically checked against the originals.

`dam_sam/static_moe.py` replaces the gate's input-dependent `argmax` with a one-hot selection
over all `M` experts, built from an `argmax` compared against `arange(M)` because the backend
accepts neither scatter nor one-hot operators. Over eight slices this changes 13 of 8.4M mask
pixels.

`dam_sam/modeling_sam_conv_lora.py` carries a `USE_SDPA` flag, on by default. It folds the
decomposed relative-position terms into a single additive bias and calls
`scaled_dot_product_attention` instead of building the attention matrix through several full-size
intermediates. Encoder outputs agree with the eager path to 1.2e-5 and the final masks are
identical. Set `USE_SDPA = False` to restore the original formulation; the adapted encoder then
exceeds the graph serializer's memory limit at 1024x1024.

## Layout

```
dam_sam/
  modeling_dam_sam.py         per-class head: frozen SAM + Conv-LoRA + mask decoder
  modeling_sam_conv_lora.py   vendored SAM with the Conv-LoRA hooks and the SDPA rewrite
  adaptation_layers.py        ConvLoRALinear and the noisy top-1 gate
  lora_inject.py              injects adapters into the fused qkv projection of every block
  losses.py                   Dice, focal, Focal-Tversky, per-image reduction
  static_moe.py               export-safe rewrite of the MoE routing
scripts/
  train_moe.py                trains both heads
  evaluate_moe.py             evaluates every configuration with fixed per-split settings
  evaluate.py                 single checkpoint, single split
  summarize_metrics*.py       formats the result tables
```
