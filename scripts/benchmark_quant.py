#!/usr/bin/env python3
"""
Deployment benchmark: full-precision vs quantized DAM-SAM.

Measures, per configuration, the six metrics that matter for live print monitoring:
  1. inference latency per image (ms) at 1024x1024
  2. throughput (img/s)
  3. peak memory footprint (VRAM allocated / reserved)
  4. model size on disk (real serialisation, pre/post quantization, pre/post LoRA merge)
  5. energy per inference (J), by integrating NVML power samples
  6. cold-start / first-inference latency

IMPORTANT -- what "quantized" means in each row:
  * ptq4sam_w8a8 / ptq4sam_w4a4 are *simulated* (fake) quantization, which is what the
    PTQ4SAM paper evaluates. Tensors are quantized then dequantized back to FP, so the
    arithmetic is still floating point. These rows are the honest accuracy configuration
    but they are SLOWER than FP -- they measure simulation cost, not deployment speed.
  * int8_real uses genuine int8 tensor-core GEMMs (torch._int_mm) with int8 weight storage.
    This is the only row whose latency/energy reflects real integer execution.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dam_sam import DamSAMHead                                  # noqa: E402
from dam_sam.transforms import SAM_IMAGE_SIZE                   # noqa: E402
from dam_sam.utils import load_joint_checkpoint                 # noqa: E402


# ───────────────────────────── energy sampling ────────────────────────────────

class PowerSampler(threading.Thread):
    """Integrates NVML power draw over the measurement window."""

    def __init__(self, interval=0.005):
        super().__init__(daemon=True)
        self.interval, self.samples = interval, []
        self._stop_ev = threading.Event()
        import pynvml
        pynvml.nvmlInit()
        self.pynvml = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def run(self):
        while not self._stop_ev.is_set():
            try:
                self.samples.append((time.perf_counter(), self.pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop_ev.set()
        self.join(timeout=2.0)

    def energy_joules(self):
        """Trapezoidal integration of the power trace."""
        if len(self.samples) < 2:
            return float("nan"), float("nan")
        e = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            e += (p1 + p0) / 2.0 * (t1 - t0)
        dur = self.samples[-1][0] - self.samples[0][0]
        return e, e / dur if dur > 0 else float("nan")


def idle_power(seconds=2.0):
    s = PowerSampler(); s.start(); time.sleep(seconds); s.stop()
    _, avg = s.energy_joules()
    return avg


# ─────────────────────────────── model builders ───────────────────────────────

def build_fp(args, device="cuda"):
    m = DamSAMHead(args.checkpoint_name, args.lora_r, args.lora_alpha, args.expert_num,
                   SAM_IMAGE_SIZE, gradient_checkpointing=False).to(device).eval()
    if args.checkpoint:
        dummy = DamSAMHead(args.checkpoint_name, args.lora_r, args.lora_alpha,
                           args.expert_num, SAM_IMAGE_SIZE, gradient_checkpointing=False)
        load_joint_checkpoint(args.checkpoint, m, dummy, map_location="cpu")
        del dummy
    return m


def patch_moe_gate_for_half():
    """
    Work around a latent dtype bug in DAM-SAM's vendored Conv-LoRA MoE gate.

    dam_sam/adaptation_layers.py:771 does
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
    The .float() pins `zeros` to fp32 while `top_k_gates` follows the model dtype, so under
    fp16/bf16 the scatter raises "Expected self.dtype to be equal to src.dtype" and half
    precision is impossible. We re-bind an eval-only forward that keeps the gate math in
    fp32 throughout. Gates are used only for expert dispatch here (multiply_by_gates=False),
    so this is numerically equivalent. Patched at benchmark time -- the model source is
    left untouched.
    """
    from dam_sam.adaptation_layers import MoEGate

    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        batch_size = feats.shape[0]
        feats_S = self.gap(feats).view(batch_size, -1)
        logits = (feats_S.float() @ self.w_gate.float())          # keep gating in fp32
        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_gates = self.softmax(top_logits[:, : self.k])
        zeros = torch.zeros_like(logits).float()
        gates = zeros.scatter(1, top_indices[:, : self.k], top_k_gates)
        load = (gates > 0).sum(0)
        loss = (self.cv_squared(gates.sum(0)) + self.cv_squared(load)) * loss_coef
        return gates, loss

    MoEGate.forward = forward


def apply_mode(model, mode, args, cali=None):
    if mode == "fp32":
        return model
    if mode in ("fp16", "bf16"):
        # Autocast rather than .half(): DAM-SAM's Conv-LoRA has several dtype assumptions
        # that break when parameters are converted outright, and autocast is in any case
        # the standard way to run mixed-precision inference.
        patch_moe_gate_for_half()
        return model
    if mode == "int8_real":
        from dam_sam.ptq4sam.realint8 import convert_to_int8
        n = convert_to_int8(model.sam)
        print(f"    [int8_real] converted {n} Linear layers to int8 tensor-core GEMM")
        return model
    if mode.startswith("ptq4sam"):
        from dam_sam.ptq4sam.calibrate import calibrate
        from dam_sam.ptq4sam.quant_sam import PTQ4SAMQuantizer
        bits = 8 if mode.endswith("w8a8") else 4
        q = PTQ4SAMQuantizer(model.sam, w_bit=bits, a_bit=bits)
        calibrate(q, cali, verbose=False)
        print(f"    [{mode}] simulated quantization active (W{bits}A{bits})")
        return model
    raise ValueError(mode)


# ─────────────────────────────── measurements ─────────────────────────────────

@torch.no_grad()
def measure(model, mode, reps=20, warmup=5, batch=1):
    import contextlib
    amp = {"fp16": torch.half, "bf16": torch.bfloat16}.get(mode)
    x = torch.randn(batch, 3, SAM_IMAGE_SIZE, SAM_IMAGE_SIZE, device="cuda", dtype=torch.float32)

    def run():
        ctx = torch.autocast("cuda", dtype=amp) if amp else contextlib.nullcontext()
        with ctx:
            return model(x)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    lat = []
    sampler = PowerSampler(); sampler.start()
    t_all = time.perf_counter()
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t_all
    sampler.stop()
    energy, avg_p = sampler.energy_joules()

    lat.sort()
    return {
        "latency_ms_median": lat[len(lat) // 2],
        "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "latency_ms_mean": sum(lat) / len(lat),
        "throughput_img_s": (reps * batch) / wall,
        "peak_vram_alloc_MB": torch.cuda.max_memory_allocated() / 2 ** 20,
        "peak_vram_reserved_MB": torch.cuda.max_memory_reserved() / 2 ** 20,
        "energy_total_J": energy,
        "energy_per_img_J": energy / (reps * batch),
        "avg_power_W": avg_p,
    }


def measure_cold_start(mode, args):
    """Fresh process: import -> build -> load weights -> first inference."""
    code = f'''
import time, sys, torch
t_start = time.perf_counter()
sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
from dam_sam import DamSAMHead
from dam_sam.transforms import SAM_IMAGE_SIZE
from dam_sam.utils import load_joint_checkpoint
t_import = time.perf_counter()
m = DamSAMHead({args.checkpoint_name!r}, {args.lora_r}, {args.lora_alpha}, {args.expert_num},
               SAM_IMAGE_SIZE, gradient_checkpointing=False).cuda().eval()
t_build = time.perf_counter()
d = DamSAMHead({args.checkpoint_name!r}, {args.lora_r}, {args.lora_alpha}, {args.expert_num},
               SAM_IMAGE_SIZE, gradient_checkpointing=False)
load_joint_checkpoint({args.checkpoint!r}, m, d, map_location="cpu"); del d
t_load = time.perf_counter()
import contextlib
_amp = torch.half if {mode!r} == "fp16" else None
def _ctx():
    return torch.autocast("cuda", dtype=_amp) if _amp else contextlib.nullcontext()
x = torch.randn(1,3,SAM_IMAGE_SIZE,SAM_IMAGE_SIZE, device="cuda", dtype=torch.float32)
with torch.no_grad(), _ctx(): m(x)
torch.cuda.synchronize()
t_first = time.perf_counter()
with torch.no_grad(), _ctx(): m(x)
torch.cuda.synchronize()
t_second = time.perf_counter()
import json; print("COLDSTART" + json.dumps({{
  "import_s": t_import-t_start, "build_s": t_build-t_import, "ckpt_load_s": t_load-t_build,
  "first_infer_s": t_first-t_load, "second_infer_s": t_second-t_first,
  "total_to_first_result_s": t_first-t_start}}))
'''
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("COLDSTART"):
            return json.loads(line[len("COLDSTART"):])
    return {"error": r.stderr[-500:]}


# ───────────────────────────── model size on disk ─────────────────────────────

def pack_int4(t: torch.Tensor) -> torch.Tensor:
    """Pack an int4 tensor (values 0..15) two-per-byte."""
    flat = t.flatten().to(torch.uint8) & 0x0F
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    return (flat[::2] | (flat[1::2] << 4)).contiguous()


def model_size_report(model, outdir):
    """Serialise real tensors at each precision and measure actual file sizes."""
    os.makedirs(outdir, exist_ok=True)
    sam = model.sam
    rows = {}

    def fsize(path):
        return os.path.getsize(path) / 2 ** 20

    sd = {k: v for k, v in sam.state_dict().items()}
    total_params = sum(v.numel() for v in sd.values())

    p = os.path.join(outdir, "fp32.pt"); torch.save(sd, p); rows["fp32"] = fsize(p)
    p = os.path.join(outdir, "fp16.pt")
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}, p)
    rows["fp16"] = fsize(p)

    # int8 / int4: quantize per output channel, store codes + scales/zeros
    for bit, name in ((8, "int8"), (4, "int4")):
        packed = {}
        for k, v in sd.items():
            if v.is_floating_point() and v.dim() >= 2:
                w = v.float().reshape(v.shape[0], -1)
                mn, mx = w.min(1)[0], w.max(1)[0]
                scale = ((mx - mn) / (2 ** bit - 1)).clamp(min=1e-12)
                zp = torch.round(-mn / scale)
                q = torch.clamp(torch.round(w / scale[:, None]) + zp[:, None], 0, 2 ** bit - 1)
                if bit == 8:
                    packed[k] = q.to(torch.uint8).reshape(v.shape)
                else:
                    packed[k] = pack_int4(q.to(torch.uint8))
                packed[k + ".scale"] = scale.half()
                packed[k + ".zp"] = zp.to(torch.uint8)
            else:
                packed[k] = v.half() if v.is_floating_point() else v
        p = os.path.join(outdir, f"{name}.pt"); torch.save(packed, p); rows[name] = fsize(p)

    return rows, total_params


def lora_merge_report(model):
    """
    Conv-LoRA mergeability.

    Vanilla LoRA folds entirely into the base weight (W += scaling * B @ A), eliminating
    the adapter. Conv-LoRA does NOT: its branch is
        result += (A x + ConvExpertMoE(A x)) @ B^T * scaling
    Only the identity term (A x) @ B^T is bilinear and thus mergeable; the MoE branch has a
    GELU, spatial convolutions, bicubic resampling and *input-dependent* top-1 gating, so it
    cannot be folded into a static weight. A and B must both be retained to feed and project
    that branch, so merging removes a matmul but frees no parameters.
    """
    from dam_sam.adaptation_layers import ConvLoRALinear
    base = lora_ab = conv_experts = gating = 0
    n_layers = 0
    for m in model.sam.modules():
        if isinstance(m, ConvLoRALinear):
            n_layers += 1
            base += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            lora_ab += m.lora_A.numel() + m.lora_B.numel()
            conv_experts += sum(p.numel() for p in m.lora_moe_experts.parameters())
            gating += sum(p.numel() for p in m.lora_moe_gating.parameters())
    total = sum(p.numel() for p in model.sam.parameters())
    return {
        "conv_lora_layers": n_layers,
        "base_weight_params_in_those_layers": base,
        "lora_A_plus_B_params": lora_ab,
        "conv_expert_params": conv_experts,
        "moe_gating_params": gating,
        "adapter_total_params": lora_ab + conv_experts + gating,
        "model_total_params": total,
        "mergeable_params": 0,
        "note": "Conv-LoRA is not fully mergeable; see docstring.",
    }


# ──────────────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/dam_sam/best.pt")
    ap.add_argument("--checkpoint_name", default="facebook/sam-vit-huge")
    ap.add_argument("--lora_r", type=int, default=2)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--expert_num", type=int, default=8)
    ap.add_argument("--modes", default="fp32,fp16,int8_real,ptq4sam_w8a8,ptq4sam_w4a4")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--calib_size", type=int, default=8)
    ap.add_argument("--datasets_root", default="datasets")
    ap.add_argument("--out", default="outputs/quant_benchmark.json")
    ap.add_argument("--size_dir", default="/tmp/dam_sam_sizes")
    ap.add_argument("--skip_cold", action="store_true")
    args = ap.parse_args()

    results = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}

    print("[BENCH] measuring idle power baseline …")
    results["idle_power_W"] = idle_power(3.0)
    print(f"        idle = {results['idle_power_W']:.2f} W")

    # calibration data for the simulated PTQ4SAM modes
    cali = None
    if any("ptq4sam" in m for m in args.modes.split(",")):
        import pandas as pd
        from PIL import Image
        from dam_sam.transforms import load_image
        root = os.path.join(args.datasets_root, "gan-generated")
        df = pd.read_csv(os.path.join(root, "train_class2.csv"))
        step = max(1, len(df) // args.calib_size)
        cali = [load_image(os.path.join(root, r["image"])).unsqueeze(0)
                for _, r in df.iloc[::step].head(args.calib_size).iterrows()]
        print(f"[BENCH] calibration set: {len(cali)} images")

    # one-off structural reports
    print("[BENCH] model size on disk …")
    m0 = build_fp(args)
    sizes, nparams = model_size_report(m0, args.size_dir)
    results["model_size_MB"] = sizes
    results["sam_state_dict_params"] = nparams
    results["lora_merge"] = lora_merge_report(m0)
    print("        " + json.dumps(sizes))
    del m0; torch.cuda.empty_cache()

    for mode in args.modes.split(","):
        print(f"\n[BENCH] === {mode} ===")
        try:
            model = build_fp(args)
            model = apply_mode(model, mode, args, cali=cali)
            r = measure(model, mode, reps=args.reps, batch=args.batch)
            results[mode] = r
            print(f"    latency {r['latency_ms_median']:.1f} ms | {r['throughput_img_s']:.2f} img/s "
                  f"| VRAM {r['peak_vram_alloc_MB']:.0f} MB | {r['energy_per_img_J']:.2f} J/img "
                  f"| {r['avg_power_W']:.1f} W")
            del model
        except Exception as e:
            import traceback; traceback.print_exc()
            results[mode] = {"error": str(e)[:300]}
        torch.cuda.empty_cache()

    if not args.skip_cold:
        for mode in ("fp32", "fp16"):
            print(f"[BENCH] cold start ({mode}) …")
            results.setdefault("cold_start", {})[mode] = measure_cold_start(mode, args)
            print("        " + json.dumps(results["cold_start"][mode]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[BENCH] wrote {args.out}")


if __name__ == "__main__":
    main()
