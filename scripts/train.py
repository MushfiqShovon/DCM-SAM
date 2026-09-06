import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dam_sam import JointAMDataset, build_loss, build_oversample_weights, per_image_loss, DamSAMHead
from dam_sam.transforms import SAM_IMAGE_SIZE
from dam_sam.utils import (
    AverageMeter,
    get_layerwise_param_groups,
    make_warmup_cosine_scheduler,
    mask_iou_dice,
    resolve_devices,
    save_joint_checkpoint,
    seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train DAM-SAM: two independent Conv-LoRA + mask-decoder heads (pore, "
        "inclusion) sharing one frozen SAM ViT-H backbone, on the same AM/XCT input batch."
    )
    p.add_argument("--dataset_root", type=str, default="datasets/gan-generated")
    p.add_argument("--train_pore", type=str, default="train_class2")
    p.add_argument("--train_incl", type=str, default="train_class3")
    p.add_argument("--val_split", type=str, default="test6", help="prefix; val CSVs are <val_split>_class2 / _class3")
    p.add_argument("--output_dir", type=str, default="checkpoints/dam_sam_vitb")
    p.add_argument("--checkpoint_name", type=str, default="facebook/sam-vit-base")
    p.add_argument("--image_size", type=int, default=SAM_IMAGE_SIZE,
                   help="Square input resolution. SAM is pretrained at 1024; smaller "
                        "values resample pos_embed and REQUIRE fine-tuning.")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=3e-4, help="base LR, applied to the deepest ViT block and the mask decoder")
    p.add_argument("--lr_decay", type=float, default=0.9, help="layerwise LR decay per ViT block toward the input end")
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--lora_r", type=int, default=2)
    p.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="scaling = lora_alpha / lora_r = 4.0 at the default rank, matching XCT-SAM's own "
        "effective scaling at r=2 (its scripts only ever override AutoGluon's optim.lora.r, "
        "leaving optim.lora.alpha at its config default of 8). Not the alpha=32 (scaling=16) "
        "the original DAM-SAM_back script defaulted to -- no rationale for that value was found "
        "anywhere in this codebase's history, and it doesn't match XCT-SAM's own precedent.",
    )
    p.add_argument("--expert_num", type=int, default=8)
    p.add_argument("--moe_loss_weight", type=float, default=0.1)

    p.add_argument("--loss_pore", type=str, default="dice_focal_loss", choices=["dice_focal_loss", "focal_tversky_loss"])
    p.add_argument("--loss_incl", type=str, default="dice_focal_loss", choices=["dice_focal_loss", "focal_tversky_loss", "bce_dice_loss"])
    p.add_argument("--tversky_alpha", type=float, default=0.3)
    p.add_argument("--tversky_beta", type=float, default=0.7)
    p.add_argument("--tversky_gamma", type=float, default=0.75)
    p.add_argument("--bce_pos_weight", type=float, default=100.0)
    p.add_argument("--bce_dice_weight", type=float, default=0.5)

    p.add_argument("--incl_oversample", type=float, default=5.0, help="WeightedRandomSampler weight for inclusion-positive images")
    p.add_argument(
        "--pore_device",
        type=str,
        default=None,
        help="defaults to cuda:0, or cuda:1 as well if --incl_device is also left unset and a second "
        "GPU is visible (auto head-parallel split, one head per GPU)",
    )
    p.add_argument(
        "--incl_device",
        type=str,
        default=None,
        help="defaults to cuda:0, or cuda:1 if left unset alongside --pore_device and a second GPU is visible",
    )

    p.add_argument("--patience", type=int, default=20, help="epochs without pore val IoU improvement before early stopping")
    p.add_argument(
        "--select_on",
        type=str,
        default="pore",
        choices=["pore", "mean"],
        help="which val IoU drives best-checkpoint selection and early stopping. The original "
        "DAM-SAM training script always tracked pore IoU only, on the reasoning that pore has "
        "the natural (non-oversampled) class distribution and is the harder head to keep "
        "stable; 'mean' averages both heads' IoU instead if you'd rather balance the two.",
    )
    p.add_argument("--seed", type=int, default=42686693)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def build_head(args, device):
    model = DamSAMHead(
        checkpoint_name=args.checkpoint_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        conv_lora_expert_num=args.expert_num,
        image_size=args.image_size,
    ).to(device)
    return model


def backward_and_step(model, optimizer, scheduler, loss, grad_clip, tag, epoch, step):
    """Skips the update (instead of applying it) when the loss is non-finite.

    Gradient clipping does not repair a non-finite gradient: if any element is already
    NaN/Inf, the computed total norm is NaN/Inf too, so the "clip" scale factor is
    NaN/Inf, and the gradient stays non-finite after "clipping." optimizer.step() then
    bakes that straight into the weights, permanently -- every later forward pass produces
    NaN activations from that point on, and the very next MoE gating call is simply the
    first place with a strict correctness check (torch.distributions.Normal.cdf) that
    raises instead of silently propagating NaN, which is what actually surfaces as a
    crash. Catching it here, before backward, is what prevents the corruption in the
    first place, rather than merely reporting it several steps later and mid-way through
    an unrelated call stack."""
    if not torch.isfinite(loss):
        print(f"[WARN] epoch {epoch} step {step}: non-finite {tag} loss ({loss.item()}), skipping this update")
        optimizer.zero_grad(set_to_none=True)
        return False
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    return True


def build_optimizer_scheduler(model, args, total_steps):
    groups = get_layerwise_param_groups(model, args.lr, args.lr_decay)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, loader, mask_key, device):
    model.eval()
    iou_meter, dice_meter = AverageMeter(), AverageMeter()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch[mask_key].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(images)
        iou, dice = mask_iou_dice(out.pred_masks.float(), masks)
        bs = images.size(0)
        iou_meter.update(iou, bs)
        dice_meter.update(dice, bs)
    model.train()
    return iou_meter.avg, dice_meter.avg


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    pore_device, incl_device = resolve_devices(args.pore_device, args.incl_device)
    print(f"[DAM-SAM] pore_device={pore_device}  incl_device={incl_device}")

    train_ds = JointAMDataset(args.dataset_root, args.train_pore, args.train_incl, image_size=args.image_size, augment=True)
    val_pore_csv = f"{args.val_split}_{args.train_pore.split('_', 1)[1]}"
    val_incl_csv = f"{args.val_split}_{args.train_incl.split('_', 1)[1]}"
    val_ds = JointAMDataset(args.dataset_root, val_pore_csv, val_incl_csv, image_size=args.image_size, augment=False)

    sample_weights = build_oversample_weights(args.dataset_root, args.train_incl, args.incl_oversample)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    pore_model = build_head(args, pore_device)
    incl_model = build_head(args, incl_device)
    print(
        f"[DAM-SAM] pore trainable: {pore_model.num_trainable_parameters() / 1e6:.2f}M  "
        f"incl trainable: {incl_model.num_trainable_parameters() / 1e6:.2f}M"
    )

    pore_loss_fn = build_loss(
        args.loss_pore, alpha=args.tversky_alpha, beta=args.tversky_beta, gamma=args.tversky_gamma
    )
    incl_loss_fn = build_loss(
        args.loss_incl,
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        gamma=args.tversky_gamma,
        pos_weight=args.bce_pos_weight,
        dice_weight=args.bce_dice_weight,
    )

    total_steps = args.epochs * len(train_loader)
    pore_optimizer, pore_scheduler = build_optimizer_scheduler(pore_model, args, total_steps)
    incl_optimizer, incl_scheduler = build_optimizer_scheduler(incl_model, args, total_steps)

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        pore_model.load_state_dict(state["pore"], strict=False)
        incl_model.load_state_dict(state["incl"], strict=False)
        start_epoch = state.get("epoch", 0) + 1
        print(f"[DAM-SAM] resumed from {args.resume} at epoch {start_epoch}")

    best_metric = -1.0
    patience = 0

    for epoch in range(start_epoch, args.epochs):
        pore_model.train()
        incl_model.train()
        loss_p_meter, loss_i_meter = AverageMeter(), AverageMeter()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"]

            # ── pore: forward + backward on pore_device ──────────────────────
            # bf16 autocast, no GradScaler: bf16 has float32's exponent range (no overflow
            # at these activation magnitudes), unlike float16, which needs loss scaling to
            # avoid underflow and can silently overflow to inf/NaN at larger batch sizes.
            # See the NaN-in-MoE-gating crash this replaced, under float16 + batch_size=8.
            pore_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_p = pore_model(images.to(pore_device, non_blocking=True))
                loss_p = pore_loss_fn(out_p.pred_masks, batch["pore_mask"].to(pore_device))
                if out_p.moe_loss is not None:
                    loss_p = loss_p + args.moe_loss_weight * out_p.moe_loss
            pore_stepped = backward_and_step(pore_model, pore_optimizer, pore_scheduler, loss_p, args.grad_clip, "pore", epoch, step)

            # ── inclusion: forward + backward on incl_device ─────────────────
            # per_image_loss, not a plain batched loss: inclusion is the sparser, heavily
            # oversampled class, and a batch-level mean lets one dense image dominate the
            # gradient for the whole step. See dam_sam.losses.per_image_loss.
            incl_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_i = incl_model(images.to(incl_device, non_blocking=True))
                loss_i = per_image_loss(incl_loss_fn, out_i.pred_masks, batch["incl_mask"].to(incl_device))
                if out_i.moe_loss is not None:
                    loss_i = loss_i + args.moe_loss_weight * out_i.moe_loss
            incl_stepped = backward_and_step(incl_model, incl_optimizer, incl_scheduler, loss_i, args.grad_clip, "incl", epoch, step)

            # Skip feeding a non-finite loss into the running-average meter too: AverageMeter
            # accumulates a running sum, so a single NaN would poison every average printed
            # for the rest of the epoch even though the update itself was correctly skipped.
            if pore_stepped:
                loss_p_meter.update(loss_p.item())
            if incl_stepped:
                loss_i_meter.update(loss_i.item())
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss_pore={loss_p_meter.avg:.4f} loss_incl={loss_i_meter.avg:.4f} "
                    f"lr={pore_optimizer.param_groups[0]['lr']:.2e}"
                )

        print(f"[epoch {epoch}] done in {(time.time() - t0) / 60:.1f} min")

        val_iou_p, val_dice_p = evaluate(pore_model, val_loader, "pore_mask", pore_device)
        val_iou_i, val_dice_i = evaluate(incl_model, val_loader, "incl_mask", incl_device)
        print(
            f"[epoch {epoch}] val: pore iou={val_iou_p:.4f} dice={val_dice_p:.4f}  "
            f"incl iou={val_iou_i:.4f} dice={val_dice_i:.4f}"
        )

        metric = val_iou_p if args.select_on == "pore" else (val_iou_p + val_iou_i) / 2
        save_joint_checkpoint(os.path.join(args.output_dir, "last.pt"), pore_model, incl_model, epoch)
        if metric > best_metric:
            best_metric = metric
            patience = 0
            save_joint_checkpoint(
                os.path.join(args.output_dir, "best.pt"), pore_model, incl_model, epoch,
                {"pore_iou": val_iou_p, "incl_iou": val_iou_i},
            )
            print(f"[epoch {epoch}] new best {args.select_on}_iou={best_metric:.4f}, checkpoint saved")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[early stop] no {args.select_on} IoU improvement for {args.patience} epochs.")
                break


if __name__ == "__main__":
    main()

# python3 scripts/train.py --dataset_root datasets/gan-generated --val_split test6 --output_dir checkpoints/dam_sam --epochs 30
#
# --pore_device/--incl_device auto-split across GPUs (cuda:0 + cuda:1) when both are left
# unset and a second GPU is visible -- no flags needed for the common two-GPU case. Pass
# either flag explicitly to pin a specific device instead (e.g. both heads on cuda:0, or a
# non-default pair like cuda:1/cuda:2).
