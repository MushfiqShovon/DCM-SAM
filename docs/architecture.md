# DAM-SAM: Architecture Deep-Dive

A reader's guide to the model your partner built — what every module does, why it is
built that way, and where the code and the design slides (`doc.pdf`) disagree. Written
against the code as it stands in this repo; line references are to the current files.

---

## 1. One paragraph

DAM-SAM turns the Segment Anything Model into a **promptless, two-class defect
segmenter** for additive-manufacturing X-ray CT slices. A frozen SAM image encoder is
adapted with **Conv-LoRA** (low-rank adapters whose middle carries a mixture of small
convolutional experts), and **two fully-trainable SAM mask decoders** — one for pores,
one for inclusions — learn to produce masks with *no prompt at all*. Out of ~631–641 M
parameters, only ~4.1 M per head (≈0.65 %) are trained: the Conv-LoRA weights and the
mask decoder. Everything else — the ViT backbone, the prompt encoder — stays frozen at
its SAM pretraining values.

**Lineage:** SAM (Kirillov et al., ICCV'23) → Conv-LoRA (Zhong et al., ICLR'24,
arXiv:2401.17868) → XCT-SAM / DAM-SAM (this repo). The backbone code is a vendored copy
of HuggingFace's SAM implementation with Conv-LoRA hooks added
(`dam_sam/modeling_sam_conv_lora.py`); the adapter code is trimmed from AutoGluon's
multimodal adaptation layers (`dam_sam/adaptation_layers.py`).

---

## 2. End-to-end data flow

```
input image  (B, 3, 1024, 1024)   ImageNet-normalized grayscale→RGB
     │
     ▼
┌─ SamVisionEncoder (frozen ViT-H, 32 blocks) ──────────────────────────┐
│  patch_embed: Conv2d k16 s16 → (B, 64, 64, 1280)   [4096 tokens]      │
│  + pos_embed (1, 64, 64, 1280)  learned absolute positions            │
│  32 × SamVisionLayer:                                                 │
│      LayerNorm → attention → +residual → LayerNorm → MLP → +residual  │
│      • blocks {7,15,23,31}: GLOBAL attention over all 4096 tokens     │
│      • all other blocks: 14×14 WINDOWED attention (5×5=25 windows)    │
│      • attn.qkv is a ConvLoRALinear  ← the ONLY adapted module        │
│      • decomposed relative position bias (rel_pos_h / rel_pos_w)      │
│  neck: 2× Conv2d + LayerNorm  → image_embeddings (B, 256, 64, 64)     │
└───────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ SamPromptEncoder (frozen, NEVER given a prompt) ─────────────────────┐
│  sparse: none                                                          │
│  dense:  no_mask_embed broadcast to (B, 256, 64, 64)                   │
│  also supplies image-wide positional encodings to the decoder          │
│  (random-Fourier SamPositionalEmbedding — see §8, quirk 2)             │
└───────────────────────────────────────────────────────────────────────┘
     │
     ▼   (run TWICE, once per head, with different trained weights)
┌─ SamMaskDecoder (fully trainable, ~3.8 M params) ─────────────────────┐
│  queries = [iou_token, 4 × mask_tokens]                                │
│  SamTwoWayTransformer (2 × SamTwoWayAttentionBlock):                   │
│      token self-attn → token→image cross-attn → MLP →                  │
│      image→token cross-attn      (+ final token→image attention)       │
│  upscale: 2 × ConvTranspose2d  (64×64 → 256×256)                       │
│  hypernetwork MLPs × mask token → low-res mask logits (B,1,256,256)    │
└───────────────────────────────────────────────────────────────────────┘
     │
     ▼
DamSAMHead: F.interpolate(bilinear) → logits (B, 1, 1024, 1024)
```

`DamSAMHead.forward` (`dam_sam/modeling_dam_sam.py`) calls the SAM model with
`multimask_output=False`, takes mask channel 0, upsamples to full resolution, and also
returns the Conv-LoRA MoE load-balancing loss.

---

## 3. The Conv-LoRA adapter (`adaptation_layers.py`)

This is the heart of the contribution the model inherits from Conv-LoRA. A plain LoRA
adapts a frozen linear layer as `y = Wx + (α/r)·B(Ax)` with `A ∈ ℝ^{r×in}`,
`B ∈ ℝ^{out×r}`. Conv-LoRA inserts a **spatial mixture-of-experts between A and B**:

```
y = Wx + (α/r) · B · [ Ax + ConvExpert_g(x)(reshape₂ᴰ(Ax)) ]
```

Step by step (`ConvLoRALinear.forward`, ~line 686):

1. `lora_res = A·x`, reshaped from tokens to a 2-D map `(B, r, H, W)` — for the vision
   encoder that is `(B, 2, 64, 64)` (or per-window sizes inside windowed blocks).
2. **Gating** (`MoEGate`): global-average-pool the map → `(B, r)` → linear to 8 logits →
   noisy **top-1** selection (Gaussian noise only during training). Exactly one expert
   runs per sample at inference.
3. **Experts**: expert *i* (i = 1…8) bicubically upsamples the map by factor *i*, applies
   `Conv2d(r→r, 3×3) + GELU`, and downsamples back. Eight experts = eight receptive-field
   scales on the low-rank map. Tiny: ~38 params each.
4. **Residual combine**: `lora_res + expert_out` (`multiply_by_gates=False` — the gate
   only routes; it does not scale).
5. Project up through `B`, scale by `α/r`, add to the frozen `Wx`.
6. Return `(result, moe_loss)` — the auxiliary loss is `cv²(importance) + cv²(load)`
   (squared coefficient of variation), pushing the gate to use all experts evenly.

`SparseDispatcher` (line 846) is the classic Shazeer-MoE bookkeeping that routes only the
selected samples through each expert. *(Deployment note: it is built on `torch.nonzero`,
`.tolist()` and data-dependent `split` — Python control flow on tensor values — which is
why the model cannot be exported to a static graph without rewriting this path; see
`docs/quantization-study/README.md` §7.)*

The file also contains `LoRALinear`, `IA3Linear`, `LoRAConv2d`, `LoRAEmbedding`,
`LoRAMergedLinear` — inherited from AutoGluon, **unused** here. Only `ConvLoRALinear`,
`MoEGate`, `SparseDispatcher` are live.

**Non-mergeability.** Unlike vanilla LoRA, this adapter cannot be folded into `W`: the
expert branch has a GELU, spatial convs and input-dependent routing. Only the identity
term `B·A` is bilinear. Merging would eliminate one matmul and zero parameters.

---

## 4. Where the adapter is injected (`lora_inject.py`) — read this carefully

```python
SAM_VISION_ENCODER_MODULE_FILTER = [r".*vision_encoder.*attn"]
SAM_VISION_ENCODER_LINEAR_FILTER = ["q", "v"]     # re.match → PREFIX match
```

The intent (and `doc.pdf` page 1) is Conv-LoRA on the **q and v projections**, as in the
original Conv-LoRA paper. But the vendored HF `SamVisionAttention` has a single **fused
`qkv` linear** (1280 → 3840). Prefix-matching `"q"` therefore hits `qkv`, `"v"` hits
nothing, and the net effect is **one ConvLoRALinear wrapping the fused qkv per attention
block** — 32 adapters total, whose low-rank update touches **q, k, AND v**.

So, contrary to `doc.pdf` page 1 ("attn.k — frozen"), **the key projection is adapted
too** in this port. Page 2 of the same slides ("Conv-LoRA in attn.qkv") reflects what the
code actually does. Nothing is wrong at runtime — it is simply a slightly *larger*
adaptation than the original design, and worth stating precisely in a paper.

Not adapted anywhere: `attn.proj`, MLP blocks, LayerNorms, patch/pos embeddings, neck,
prompt encoder.

---

## 5. Two heads, one task each (`modeling_dam_sam.py`, `train.py`)

DAM-SAM trains **one complete `DamSAMHead` per defect class** rather than one shared
adapter with two output branches:

- each head = frozen SAM + its own 32 Conv-LoRA adapters + its own mask decoder;
- completely separate forward passes, losses, optimizers, LR schedules, and gradient
  clipping (`doc.pdf` pages 2/4: "gradient isolation");
- rationale (docstring in `modeling_dam_sam.py`): pores and inclusions differ enough in
  scale, density and false-positive risk that dedicated capacity is worth the ~2×
  trainable-parameter cost — the opposite trade from BaT-SAM's single shared adapter.

**Doc-vs-code note:** the slides say the ViT-H backbone is "loaded once, shared across
both heads." In this code, each `DamSAMHead` calls `SamModel.from_pretrained` itself, so
`train.py`/`evaluate.py` hold **two full frozen backbone copies in memory**. Numerically
identical; memory-wise ~2× the backbone footprint (the auto two-GPU split in
`resolve_devices` puts one head per GPU when two are visible, which is presumably why).

**Promptless operation:** the SAM prompt encoder is kept but never prompted. Its learned
"no prompt" defaults (the dense `no_mask_embed`, plus the image-wide positional encoding
it computes for the decoder) are used as-is; the decoder must localize defects from the
adapted image embedding alone. This is the "automatic" mode of the original
DAM-SAM/XCT-SAM scripts.

### Parameter budget (measured, ViT-H, r=2, α=8, 8 experts)

| Component | Params | Trainable? |
|---|---|---|
| Vision encoder (incl. frozen qkv, MLPs, norms) | ~637.4 M | ✗ (except LoRA) |
| — Conv-LoRA `lora_A`+`lora_B` (32 layers) | 327,680 | ✓ |
| — Conv experts (32 × 8 × Conv3×3 on r=2) | 9,728 | ✓ |
| — MoE gates (`w_gate`+`w_noise`) | 1,024 | ✓ |
| Prompt encoder | ~6 K | ✗ |
| Mask decoder (transformer + upscaling + hypernet + IoU head) | ~3.80 M | ✓ |
| **Trainable per head** | **4.14 M (0.65 %)** | |
| **Total per head** | ~641.4 M | |

Freezing is enforced by name in `DamSAMHead._set_trainable`: `requires_grad ⇔` name
contains `lora_` or starts with `mask_decoder`. Checkpoints store **only** those
trainable tensors (`trainable_state_dict`, ~35 MB for both heads), on the premise that
the frozen backbone reloads identically from `checkpoint_name` (see §8 quirk 2 for the
one buffer that premise misses).

---

## 6. Training pipeline (`train.py`, `dataset.py`, `losses.py`, `transforms.py`)

**Data.** `JointAMDataset` reads paired `train_class2.csv` (pore) / `train_class3.csv`
(inclusion) that must reference identical images row-for-row; each item returns the image
plus *both* masks so one batch feeds both heads. Images: grayscale → RGB, bilinear resize
to 1024², ImageNet normalization. Masks: nearest resize, binarize > 127.

**Imbalance & augmentation** (`doc.pdf` p5, matches code): inclusion-positive images
oversampled 5× via `WeightedRandomSampler`; joint augmentation = h-flip p=.5, v-flip
p=.5, one of 90/180/270° rotation p=.5, applied identically to image and both masks.

**Losses** (`losses.py`). Default for both heads: `DiceFocalLoss` (equal-weight Dice +
focal α=.25 γ=2). Also available: `FocalTverskyLoss` (α=.3 β=.7 γ=.75 — punishes missed
defects harder than false alarms) and `BCEDiceLoss` (pos_weight=100, for ultra-sparse
targets). The inclusion loss is wrapped in `per_image_loss`, averaging per image so one
dense oversampled image cannot dominate a batch-level mean. Total loss adds
`0.1 × moe_loss` per head.

The loss file also documents two hard-won NaN fixes worth citing as engineering detail:
(a) focal loss's `(1-p_t)^γ` has a NaN *gradient* at exactly `p_t=1` (torch computes the
pow backward as `γ·result/base` → 0/0), which routine perfectly-confident background
pixels hit — clamped to 1e-7; (b) Focal-Tversky's `(1-TI)^γ` has an infinite derivative
at 0 for γ<1 — same clamp.

**Optimization.** Per head: AdamW, weight-decay 1e-3, **layer-wise LR decay** — the mask
decoder and deepest ViT block get `lr=3e-4`, each earlier block ×0.9
(`get_layerwise_param_groups`; the block count is inferred from the model — it was
hardcoded to ViT-H's 32 until the ViT-B port fixed it). Schedule: 10 % linear warmup
(from 1 %) then cosine to 0, stepped per *step*. Grad-clip max-norm 1.0.

**Precision.** `bfloat16` autocast, **no GradScaler**. The code comments record the
history: the doc's fp16+GradScaler setup (`doc.pdf` p4) NaN'd at batch 8 inside the MoE
gate; bf16 has fp32's exponent range and needs no scaler. A pre-backward
`torch.isfinite(loss)` guard skips any non-finite update entirely — the comment explains
that clipping cannot repair a non-finite gradient and the corruption otherwise surfaces
crashes later, deep inside `Normal.cdf` in the gating.

**Selection & stopping.** Validation on `test6` every epoch (bf16, threshold 0.5,
`mask_iou_dice`). `--select_on pore` (default, per the slides: pore has the natural class
distribution and is the harder head) or `mean`. `best.pt`/`last.pt` hold
`{pore, incl, epoch, pore_iou, incl_iou}` with trainable-only state dicts. Early stop:
patience 20. Gradient checkpointing on the vision encoder is enabled by default.

**`train_moe.py`** is an expert-count ablation harness: identical training, but per-head
`--expert_num_pore/--expert_num_incl` and **per-head** best-checkpoint tracking (so an
inclusion sweep isn't scored on weights selected by pore IoU), with runs laid out as
`p{Np}_i{Ni}/` for a 1/2/4/8 sweep grid.

---

## 7. Evaluation pipeline (`evaluate.py`, `summarize_metrics.py`)

Per split, per head, in order:

1. **Preprocess**: optional CLAHE (adaptive hist-eq) and gamma correction on the
   grayscale before RGB conversion; resize to model resolution; ImageNet normalize.
2. **Predict** (`predict_mask`): bf16 forward; optional **TTA** (average sigmoid over
   identity + h-flip + v-flip); bilinear upsample of *probabilities* to native
   resolution; threshold (per-class, per-split CLI flags).
3. **Morphology**: binary opening with a `morph_size²` structuring element.
4. **Blob filter** (`filter_components`): drop connected components smaller than
   `min_blob` or larger than `max_blob` pixels.
5. **Valid region**: Otsu-threshold the grayscale, fill, fit a minimum enclosing circle —
   the specimen disk; predictions *and* GT outside it are zeroed (XCT slices are circular
   cross-sections; outside is air). Alternative `nonzero` mode; `--invert_pore` exists
   for splits where the model segments solid instead of pores.
6. **Metrics**, all at native resolution: IoU, Dice, and **tolerance-F1** (a predicted
   pixel counts as TP within 5 px of any GT pixel — via mutual binary dilation —
   forgiving boundary offsets that sink strict IoU on few-pixel defects). Reported
   split by defect-bearing vs empty images: `IoU(pos)` and TNR. **Caution:** empty
   images score IoU = 1.0, so `Mean IoU` is inflated on sparse splits — `IoU(pos)` is
   the honest number.
7. Five-panel figures (input | pore GT | incl GT | pore pred | incl pred) and an
   `inf_metrics.txt` per split; `summarize_metrics.py` tabulates across splits.
8. `scripts/run.sh` pins per-split thresholds/morphology/blob bounds **tuned for the
   ViT-H@1024 checkpoint** — they do not transfer to other backbones or resolutions
   (measured: they zero out a working inclusion head; see quantization-study §12.5).

---

## 8. Quirks and known issues (verified in this repo)

1. **fp16 is broken.** `adaptation_layers.py:771` pins the gate-scatter buffer to fp32
   (`zeros_like(logits).float()`), so `model.half()` crashes in `scatter()`. Use
   `torch.autocast` instead (bf16 training does; the GPU benchmark measured FP16
   autocast at 1.7× FP32).
2. **One unsaved random buffer.** `SamPositionalEmbedding.positional_embedding` is
   `randn` at construction, absent from both the HF checkpoint and `best.pt` (it is not
   trainable), and *is* used promptlessly (image-wide positional encodings). `train.py`
   seeds; `evaluate.py` does not → eval is non-deterministic. Measured impact ≈ 0.0002
   IoU — cosmetic, but a one-line seed away from bit-reproducible.
3. **Softmax pinned to fp32** in vision attention (`dtype=torch.float32`) — irrelevant on
   GPU, but it materializes a 1 GB tensor in any half-precision NPU graph.
4. **Fixed 1024² input** — `pos_embed` is a 64×64 grid and the patch embed validates the
   size. The `--image_size` flag added during this project resamples `pos_embed` and
   patches the prompt-encoder grid (`DamSAMHead._adapt_input_resolution`); `rel_pos`
   self-interpolates. Any other resolution requires re-finetuning.
5. **No requirements file**; the vendored SAM requires `transformers<5`
   (`_tied_weights_keys` list-vs-dict).

## 9. Doc-vs-code discrepancy table (for the paper)

| Topic | `doc.pdf` says | Code does |
|---|---|---|
| Adapter placement | separate wrappers on `attn.q`, `attn.v`; `k` frozen (p1) | one wrapper on fused `qkv` → q, **k**, v all adapted (p2 of the slides agrees) |
| LoRA α | 32 (p3) | **8** (default; `train.py` argues α=32 had no traceable rationale and α/r=4 matches XCT-SAM) |
| Backbone sharing | single backbone shared by both heads (p3) | two full copies, one per `DamSAMHead` |
| Loss | Focal-Tversky (p2) | `dice_focal_loss` default (Focal-Tversky available via flag) |
| Precision | fp16 + per-head GradScaler (p4) | bf16 autocast, no scaler (fp16 NaN'd; see comments) |
| Scheduler | CosineAnnealingLR, η_min = lr·0.01 (p6) | 10 % linear warmup → cosine to 0 |
| Checkpoint name | `joint_model.ckpt` (p6) | `best.pt` / `last.pt` |
| Trainable count | ~3.97 M/head (p3) | 4.14 M/head measured (α/r and settings differ) |

Pattern: the slides describe an earlier iteration (the "DAM-SAM_back" script that
`train.py`'s comments reference); the current code is a cleaned re-implementation with
several stability fixes. Matches between doc and code: freezing scheme, 5× inclusion
oversampling, augmentation set, patience-20 early stop on pore IoU, per-head optimizers
and clipping, 1024² preprocessing, Otsu valid-region protocol.

## 10. File map

| File | Role |
|---|---|
| `dam_sam/modeling_sam_conv_lora.py` | vendored HF SAM (encoder / prompt encoder / two-way decoder) with Conv-LoRA return-tuple hooks |
| `dam_sam/adaptation_layers.py` | `ConvLoRALinear` + `MoEGate` + `SparseDispatcher` (plus unused AutoGluon LoRA variants) |
| `dam_sam/lora_inject.py` | regex-driven in-place replacement of `qkv` with `ConvLoRALinear` |
| `dam_sam/modeling_dam_sam.py` | `DamSAMHead`: promptless wrapper, freezing, resolution adaptation, trainable-only checkpoints |
| `dam_sam/dataset.py` | joint pore+incl dataset, oversample weights |
| `dam_sam/losses.py` | Dice+Focal / Focal-Tversky / BCE-Dice, per-image averaging, NaN clamps |
| `dam_sam/transforms.py` | load/normalize, joint augmentation, Otsu valid region |
| `dam_sam/utils.py` | seeding, layer-wise LR groups, warmup-cosine, IoU/Dice, tolerance-F1, joint checkpoint I/O |
| `scripts/train.py` | two-head trainer (bf16, isolation, early stop) |
| `scripts/train_moe.py` | expert-count ablation trainer |
| `scripts/evaluate.py` | full inference + post-processing + metrics + panels |
| `scripts/summarize_metrics.py` | cross-split results tables |
| `doc.pdf` | partner's 6-page architecture slides (see §9 caveats) |

*Added during the experiments in this project (not the original architecture):*
`dam_sam/ptq4sam/`, `scripts/quantize_ptq4sam.py`, `scripts/benchmark_quant.py`,
`scripts/run_vitb.sh`, `scripts/run_vith512.sh`, `docs/quantization-study/`, `paper/`.
