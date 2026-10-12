# DCM-SAM

Reference implementation for *DCM-SAM: Defect-Conditioned Mixture of LoRA Experts for
NPU-Deployed AM Defect Segmentation*.

The Python package is named `dam_sam/` and checkpoint directories are named `dam_sam_*`; these
are the original module paths and are kept so the code runs as written.

One frozen Segment Anything backbone hosts a separate Conv-LoRA expert bank and mask decoder for
each defect class. The two heads are trained without prompts, each with its own optimizer on a shared
stream of batches, and the whole
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

## Data

Two datasets are used, and neither is redistributed here.

**Synthetic training and test slices.** The CycleGAN-synthesised XCT slices come from
[SAM-Material-GAN](https://github.com/anikat1/SAM-Material-GAN/tree/main/datasets). That
repository keeps its CSVs one level above the image folders; move them inside
`gan-generated/` so the layout matches what the scripts expect. The CSV paths are already
relative to `gan-generated/`, so nothing else changes.

**Real NIST scans.** The out-of-distribution pore evaluation uses the NIST CoCr AM XCT data,
available from
[NIST](https://www.nist.gov/el/intelligent-systems-division-73500/cocr-am-xct-data). The
evaluation reads five specimen splits, `test2` to `test6`, with pore labels only, in the same
CSV format as the synthetic data.

```
datasets/
  gan-generated/
    train/ test1/ ... test6/       each with input/, class_1/, class_2/, class_3/
    <split>_class2.csv             pore masks      (columns: image, label)
    <split>_class3.csv             inclusion masks
  nist/
    test2/ ... test6/              each with input/ and class_2/
    <split>_class2.csv             pore masks only
```

Paths in each CSV are relative to the dataset folder that holds it. The synthetic `test4` and
`test5` splits carry pore labels only, and `test3` is not used.

## Checkpoints

Trained weights are not stored in git. The ViT-B runs, including the deployed `(2,8)` model,
are in `dam_sam_vitb_moe.zip` on
[Google Drive](https://drive.google.com/drive/folders/16Qbibw45K3zoqXRDAnTPIhskPuMqiBuQ).
Unzip it inside `checkpoints/`; each run then sits at
`checkpoints/dam_sam_vitb_moe/<run_tag>/best.pt` next to its `train_config.json`, which is
where `run_moe_vitb.sh` looks for it.

Every training run is single-seed. Retraining from the configs below reproduces the reported
numbers to within GPU nondeterminism: a fresh run of the deployed model gave 36.0 / 48.5
pore / inclusion IoU against the reported 36.1 / 48.9.

## Training

One head per defect class, each with its own expert count. Both heads see the same oversampled
stream of batches but have separate optimizers, schedules and gradient clipping. Only the
Conv-LoRA parameters and the mask decoders are updated; the backbone and prompt encoder stay
frozen.

The exact configuration of every reported run is in `configs/`, copied from the
`train_config.json` that training writes next to each checkpoint. Any value not passed on the
command line below already equals the script default. **Batch size is the one setting that
differs between the two backbones, and the script default (2) is not the ViT-B setting.**

The deployed model, ViT-B with two pore experts and eight inclusion experts:

```bash
python3 scripts/train_moe.py \
    --checkpoint_name facebook/sam-vit-base \
    --expert_num_pore 2 --expert_num_incl 8 --batch_size 8 \
    --output_dir checkpoints/dam_sam_vitb_moe --run_tag p2_i8 \
    --epochs 30 --val_split test6
```

The ViT-B ablation repeats this for every configuration in `configs/vitb/`:

```bash
for tag in p1_i1 p1_i8 p2_i2 p2_i8 p4_i8 p8_i1 p8_i2 p8_i4 p8_i8; do
  P=${tag#p}; P=${P%_i*}; I=${tag#*_i}
  python3 scripts/train_moe.py --checkpoint_name facebook/sam-vit-base \
      --expert_num_pore $P --expert_num_incl $I --batch_size 8 \
      --output_dir checkpoints/dam_sam_vitb_moe --run_tag $tag --epochs 30 --val_split test6
done
```

The ViT-H expert-count study (`configs/vith/`) uses the same loop with
`--checkpoint_name facebook/sam-vit-huge --batch_size 2 --output_dir checkpoints/dam_sam_moe`
over `p1_i1 p1_i8 p2_i1 p2_i8 p4_i8 p8_i1 p8_i2 p8_i4 p8_i8`.

The two ViT-H models in the resolution comparison use the single-expert-count trainer,
`scripts/train.py`, with eight experts per head:

```bash
python3 scripts/train.py --checkpoint_name facebook/sam-vit-huge --output_dir checkpoints/dam_sam
python3 scripts/train.py --checkpoint_name facebook/sam-vit-huge --image_size 512 \
    --output_dir checkpoints/dam_sam_vith_512
```

These two were trained before the sampler-seeding fix described in the paper's appendix, so a
retrain with the current script will not match them exactly. They share one configuration, so
the comparison between them is unaffected.

`--val_split test6` is also one of the reported test splits, so the five-split means include
the selection split.

## Evaluation

Every configuration is scored with a fixed per-split post-processing configuration that
`scripts/evaluate_moe.py` hardcodes in `SPLIT_CONFIGS`, applied identically to every model so the
comparison isolates the model rather than the threshold. `evaluate_moe.py` reads each run's
`train_config.json` to build heads with the right expert counts.

```bash
# ViT-B: all configurations, all synthetic splits, plus the ablation summary
bash scripts/run_moe_vitb.sh

# ViT-B: the same on the real NIST specimens
python3 scripts/evaluate_moe.py --checkpoint_name facebook/sam-vit-base \
    --checkpoints_root checkpoints/dam_sam_vitb_moe --outputs_root outputs/dam_sam_vitb_moe \
    --splits nist/test2 nist/test3 nist/test4 nist/test5 nist/test6

# one configuration only
python3 scripts/evaluate_moe.py --checkpoint_name facebook/sam-vit-base \
    --checkpoints_root checkpoints/dam_sam_vitb_moe --outputs_root outputs/dam_sam_vitb_moe \
    --runs p2_i8 --splits gan-generated/test1

# ViT-H expert-count study, and the two ViT-H resolution models
bash scripts/run_moe.sh
bash scripts/run.sh
bash scripts/run_vith512.sh            # 512x512, the 1024x1024 post-processing as is
bash scripts/run_vith512.sh scaled     # 512x512, blob and morphology sizes rescaled
```

Results land in `outputs/<root>/<run_tag>/<dataset>/<split>/inf_metrics.txt`. Use
`python3 scripts/summarize_metrics_moe.py outputs/dam_sam_vitb_moe` for the cross-run tables and
`python3 scripts/summarize_metrics.py outputs/dam_sam_vitb_moe/p2_i8` for one run.

`scripts/evaluate.py` builds both heads with the same expert count (`--expert_num`, default 8),
so use it only for checkpoints from `train.py`; MoE checkpoints with unequal counts go through
`evaluate_moe.py`.

IoU is averaged over images, and an image with no defect and no predicted pixel scores 1. The
per-split reports also print `IoU(pos)`, the average over defect-bearing images only, which is the
number to read for segmentation quality on sparse splits.

## Where each result comes from

| Paper result | Produced by |
|---|---|
| ViT-B configurations (synthetic and NIST columns) | ViT-B ablation training, then `run_moe_vitb.sh` and the NIST command above |
| Comparison with XCT-SAM baselines | DCM-SAM row: `p2_i8` from the same runs; the defect-only row is `IoU(pos)`. Baseline rows are quoted from XCT-SAM and are not reproduced here |
| Resolution against backbone | `run.sh` (ViT-H, 1024), `run_vith512.sh` and `run_vith512.sh scaled` (ViT-H, 512), `p2_i8` (ViT-B) |
| ViT-H expert-count study | ViT-H ablation training, then `run_moe.sh` |
| Per-split NIST results | `p2_i8` rows of the NIST evaluation |
| Static MoE and SDPA equivalence | `dam_sam/static_moe.py` and the `USE_SDPA` flag below |

The on-device compilation and profiling results (memory limits, latency, device-side mask
agreement) were produced with Qualcomm AI Hub; those scripts are not part of this repository.

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
  train_moe.py                trains both heads, each with its own expert count
  train.py                    trains both heads with one shared expert count
  evaluate_moe.py             evaluates every configuration with fixed per-split settings
  evaluate.py                 single checkpoint, single split
  run*.sh                     evaluation entry points for each set of results
  summarize_metrics*.py       formats the result tables
configs/
  vitb/, vith/                training configuration of every reported run
```
