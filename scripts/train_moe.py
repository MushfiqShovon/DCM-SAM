"""
scripts/train_moe.py
────────────────────────────────────────────────────────────────────────────────
MoE-expert-count ablation training for DAM-SAM.

Identical to scripts/train.py except for the two changes this ablation needs:

1. Per-head expert counts: --expert_num_pore / --expert_num_incl replace the
   single --expert_num, so the two sweeps can vary one head while holding the
   other fixed at the default (8).

2. Per-head best-checkpoint selection: train.py selects best.pt on pore val IoU
   (--select_on pore), which couples the inclusion head's saved weights to pore
   performance — fine for the main model, but it would confound the inclusion
   sweep (runs varying expert_num_incl would be scored on inclusion weights
   chosen by pore IoU). Here each head tracks its own best epoch independently:
   best.pt always holds the pore weights from the best-pore-IoU epoch and the
   inclusion weights from the best-incl-IoU epoch. Early stopping fires only
   when BOTH heads have gone --patience epochs without improvement.

3. Data-pipeline RNG decoupled from model size: train.py (and the first version
   of this script) let the WeightedRandomSampler and DataLoader worker seeding
   fall back to the global default torch generator. pore_model is built before
   incl_model, so pore's own weight init is unaffected by expert_num_incl -- but
   incl_model's Conv2d expert layers consume a variable number of draws from
   that SAME global generator during construction, which shifts the generator's
   state before the sampler/DataLoader are first iterated. Net effect: two runs
   with an IDENTICAL pore architecture (e.g. expert_num_pore=1 paired with
   expert_num_incl=1 vs. 8) can train on different sample orders and end up with
   different pore IoU, purely because the OTHER head's size changed -- observed
   directly in this ablation (p1_i1 pore IoU 0.378 vs. p1_i8 pore IoU 0.328, a
   larger gap than the entire expert_num_pore=2..8 range). Fixed by giving the
   sampler and train DataLoader their own generators, each seeded with
   torch.Generator().manual_seed(args.seed) directly from the CLI seed rather
   than drawn from the default generator's post-model-construction state, plus a
   worker_init_fn that reseeds Python's random/NumPy per worker (dam_sam.
   transforms.joint_augment's flip/rotate augmentation uses the stdlib random
   module, which PyTorch's default per-worker reseeding does not touch).

Checkpoints go to <output_dir>/<run_tag>/ where run_tag defaults to
"p{expert_num_pore}_i{expert_num_incl}", so the 8 ablation runs never collide:

    checkpoints/dam_sam_moe/
        p1_i8/  p2_i8/  p4_i8/  p8_i8/      (pore sweep, incl fixed at 8)
        p8_i1/  p8_i2/  p8_i4/              (incl sweep, pore fixed at 8)
        p1_i1/                              (corner run, interaction check)

All other settings (losses, LR schedule, oversampling, seed, bf16, non-finite
skip) are unchanged from scripts/train.py so the sweep isolates expert count.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
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
        description="MoE expert-count ablation for DAM-SAM: scripts/train.py with per-head "
        "expert counts and per-head best-checkpoint selection."
    )
    p.add_argument("--dataset_root", type=str, default="datasets/gan-generated")
    p.add_argument("--train_pore", type=str, default="train_class2")
    p.add_argument("--train_incl", type=str, default="train_class3")
    p.add_argument("--val_split", type=str, default="test6", help="prefix; val CSVs are <val_split>_class2 / _class3")
    p.add_argument("--output_dir", type=str, default="checkpoints/dam_sam_vitb_moe")
    p.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="subfolder of --output_dir for this run; defaults to p{expert_num_pore}_i{expert_num_incl}",
    )
    p.add_argument("--checkpoint_name", type=str, default="facebook/sam-vit-base")

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
    p.add_argument("--lora_alpha", type=int, default=8, help="scaling = lora_alpha / lora_r = 4.0 at the default rank; see scripts/train.py")
    p.add_argument("--expert_num_pore", type=int, default=8, help="Conv-LoRA experts in the pore head")
    p.add_argument("--expert_num_incl", type=int, default=8, help="Conv-LoRA experts in the inclusion head")
    p.add_argument("--moe_loss_weight", type=float, default=0.1)

    p.add_argument("--loss_pore", type=str, default="dice_focal_loss", choices=["dice_focal_loss", "focal_tversky_loss"])
    p.add_argument("--loss_incl", type=str, default="dice_focal_loss", choices=["dice_focal_loss", "focal_tversky_loss", "bce_dice_loss"])
    p.add_argument("--tversky_alpha", type=float, default=0.3)
    p.add_argument("--tversky_beta", type=float, default=0.7)
    p.add_argument("--tversky_gamma", type=float, default=0.75)
    p.add_argument("--bce_pos_weight", type=float, default=100.0)
    p.add_argument("--bce_dice_weight", type=float, default=0.5)

    p.add_argument("--incl_oversample", type=float, default=5.0, help="WeightedRandomSampler weight for inclusion-positive images")
    p.add_argument("--pore_device", type=str, default=None, help="see scripts/train.py — auto head-parallel split when both are unset")
    p.add_argument("--incl_device", type=str, default=None)

    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="epochs without val IoU improvement before a head is considered converged; "
        "training stops only when BOTH heads are past patience",
    )
    p.add_argument("--seed", type=int, default=42686693)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def build_head(args, device, expert_num):
    model = DamSAMHead(
        checkpoint_name=args.checkpoint_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        conv_lora_expert_num=expert_num,
        image_size=SAM_IMAGE_SIZE,
    ).to(device)
    return model


def backward_and_step(model, optimizer, scheduler, loss, grad_clip, tag, epoch, step):
    """Skips the update when the loss is non-finite — see scripts/train.py for why clipping
    cannot repair an already-NaN gradient and skipping before backward is the real fix."""
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


def cpu_state_copy(model):
    """Detached CPU snapshot of the trainable delta, safe to hold across epochs while the
    live model keeps training."""
    return {k: v.detach().cpu().clone() for k, v in model.trainable_state_dict().items()}


class HeadTracker:
    """Independent best-IoU tracking for one head: keeps a CPU snapshot of the trainable
    weights from that head's own best-val-IoU epoch, plus its own patience counter."""

    def __init__(self, tag):
        self.tag = tag
        self.best_iou = -1.0
        self.best_epoch = -1
        self.best_state = None
        self.stale_epochs = 0

    def update(self, model, iou, epoch):
        if iou > self.best_iou:
            self.best_iou = iou
            self.best_epoch = epoch
            self.best_state = cpu_state_copy(model)
            self.stale_epochs = 0
            return True
        self.stale_epochs += 1
        return False


def save_best_checkpoint(path, pore_tracker, incl_tracker):
    """Same pore/incl key layout as save_joint_checkpoint, but each half comes from its own
    head's best epoch rather than one jointly-selected epoch."""
    torch.save(
        {
            "pore": pore_tracker.best_state,
            "incl": incl_tracker.best_state,
            "epoch": max(pore_tracker.best_epoch, incl_tracker.best_epoch),
            "pore_epoch": pore_tracker.best_epoch,
            "incl_epoch": incl_tracker.best_epoch,
            "pore_iou": pore_tracker.best_iou,
            "incl_iou": incl_tracker.best_iou,
        },
        path,
    )


def worker_init_fn(worker_id):
    """Reseeds Python's random and NumPy per DataLoader worker from that worker's own torch
    seed. PyTorch's default worker seeding only reseeds torch's own per-worker generator
    (from base_seed + worker_id); Python's random module -- used by dam_sam.transforms.
    joint_augment for flip/rotate augmentation -- and NumPy are not reseeded per worker
    without this, so they'd otherwise just inherit whatever state the forked parent
    process happened to be in."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    args = parse_args()
    run_tag = args.run_tag or f"p{args.expert_num_pore}_i{args.expert_num_incl}"
    run_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    seed_everything(args.seed)

    with open(os.path.join(run_dir, "train_config.json"), "w") as fh:
        json.dump(vars(args) | {"run_tag": run_tag}, fh, indent=2)

    pore_device, incl_device = resolve_devices(args.pore_device, args.incl_device)
    print(f"[MoE-ablation:{run_tag}] pore_device={pore_device}  incl_device={incl_device}")
    print(f"[MoE-ablation:{run_tag}] expert_num_pore={args.expert_num_pore}  expert_num_incl={args.expert_num_incl}")

    train_ds = JointAMDataset(args.dataset_root, args.train_pore, args.train_incl, image_size=SAM_IMAGE_SIZE, augment=True)
    val_pore_csv = f"{args.val_split}_{args.train_pore.split('_', 1)[1]}"
    val_incl_csv = f"{args.val_split}_{args.train_incl.split('_', 1)[1]}"
    val_ds = JointAMDataset(args.dataset_root, val_pore_csv, val_incl_csv, image_size=SAM_IMAGE_SIZE, augment=False)

    # Dedicated generators, manual_seed()'d straight from args.seed rather than drawn from
    # the default global generator -- keeps sample order and worker seeding invariant to
    # expert_num_pore/expert_num_incl (which otherwise perturb the global generator's state
    # by however many Conv2d experts get initialized). See the module docstring, point 3.
    sampler_generator = torch.Generator().manual_seed(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed + 1)

    sample_weights = build_oversample_weights(args.dataset_root, args.train_incl, args.incl_oversample)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True, generator=sampler_generator
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, generator=loader_generator, worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    pore_model = build_head(args, pore_device, args.expert_num_pore)
    incl_model = build_head(args, incl_device, args.expert_num_incl)
    # Per-head trainable counts differ once expert counts differ — report both, plus the
    # delta, for the ablation table's parameter column.
    pore_params = pore_model.num_trainable_parameters()
    incl_params = incl_model.num_trainable_parameters()
    print(
        f"[MoE-ablation:{run_tag}] pore trainable: {pore_params / 1e6:.3f}M ({args.expert_num_pore} experts)  "
        f"incl trainable: {incl_params / 1e6:.3f}M ({args.expert_num_incl} experts)"
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
        print(f"[MoE-ablation:{run_tag}] resumed from {args.resume} at epoch {start_epoch} "
              f"(per-head best trackers restart from scratch)")

    pore_tracker = HeadTracker("pore")
    incl_tracker = HeadTracker("incl")

    for epoch in range(start_epoch, args.epochs):
        pore_model.train()
        incl_model.train()
        loss_p_meter, loss_i_meter = AverageMeter(), AverageMeter()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"]

            # bf16 autocast, no GradScaler — see scripts/train.py for the float16 NaN history.
            pore_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_p = pore_model(images.to(pore_device, non_blocking=True))
                loss_p = pore_loss_fn(out_p.pred_masks, batch["pore_mask"].to(pore_device))
                if out_p.moe_loss is not None:
                    loss_p = loss_p + args.moe_loss_weight * out_p.moe_loss
            pore_stepped = backward_and_step(pore_model, pore_optimizer, pore_scheduler, loss_p, args.grad_clip, "pore", epoch, step)

            # per_image_loss for inclusion — see scripts/train.py and dam_sam.losses.
            incl_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_i = incl_model(images.to(incl_device, non_blocking=True))
                loss_i = per_image_loss(incl_loss_fn, out_i.pred_masks, batch["incl_mask"].to(incl_device))
                if out_i.moe_loss is not None:
                    loss_i = loss_i + args.moe_loss_weight * out_i.moe_loss
            incl_stepped = backward_and_step(incl_model, incl_optimizer, incl_scheduler, loss_i, args.grad_clip, "incl", epoch, step)

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

        save_joint_checkpoint(os.path.join(run_dir, "last.pt"), pore_model, incl_model, epoch)

        pore_improved = pore_tracker.update(pore_model, val_iou_p, epoch)
        incl_improved = incl_tracker.update(incl_model, val_iou_i, epoch)
        if pore_improved or incl_improved:
            save_best_checkpoint(os.path.join(run_dir, "best.pt"), pore_tracker, incl_tracker)
            improved = " + ".join(
                f"{t.tag}_iou={t.best_iou:.4f}" for t, i in ((pore_tracker, pore_improved), (incl_tracker, incl_improved)) if i
            )
            print(f"[epoch {epoch}] new best {improved}, best.pt updated "
                  f"(pore@{pore_tracker.best_epoch}, incl@{incl_tracker.best_epoch})")

        if pore_tracker.stale_epochs >= args.patience and incl_tracker.stale_epochs >= args.patience:
            print(f"[early stop] neither head improved for {args.patience} epochs "
                  f"(pore stale {pore_tracker.stale_epochs}, incl stale {incl_tracker.stale_epochs}).")
            break

    print(
        f"[MoE-ablation:{run_tag}] finished. best pore iou={pore_tracker.best_iou:.4f} (epoch {pore_tracker.best_epoch})  "
        f"best incl iou={incl_tracker.best_iou:.4f} (epoch {incl_tracker.best_epoch})  "
        f"checkpoint: {os.path.join(run_dir, 'best.pt')}"
    )


if __name__ == "__main__":
    main()

# The 8 ablation runs (defaults: dataset_root=datasets/gan-generated, val_split=test6,
# output_dir=checkpoints/dam_sam_moe; run_tag is derived automatically):
#
#   # pore sweep (incl fixed at 8)
#   python3 scripts/train_moe.py --expert_num_pore 1 --expert_num_incl 8
#   python3 scripts/train_moe.py --expert_num_pore 2 --expert_num_incl 8
#   python3 scripts/train_moe.py --expert_num_pore 4 --expert_num_incl 8
#   python3 scripts/train_moe.py --expert_num_pore 8 --expert_num_incl 8   # shared anchor
#
#   # inclusion sweep (pore fixed at 8)
#   python3 scripts/train_moe.py --expert_num_pore 8 --expert_num_incl 1
#   python3 scripts/train_moe.py --expert_num_pore 8 --expert_num_incl 2
#   python3 scripts/train_moe.py --expert_num_pore 8 --expert_num_incl 4
#
#   # corner run (interaction check)
#   python3 scripts/train_moe.py --expert_num_pore 1 --expert_num_incl 1
#
# Checkpoints land in checkpoints/dam_sam_moe/<run_tag>/best.pt (pore half from the
# best-pore-IoU epoch, incl half from the best-incl-IoU epoch — evaluate.py loads it like
# any joint checkpoint). Point the future evaluation_moe.py at outputs/dam_sam_moe/<run_tag>/.
