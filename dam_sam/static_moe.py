"""
Static (export-safe) rewrite of Conv-LoRA's MoE routing.

The training-time implementation routes samples through SparseDispatcher, which uses
torch.nonzero, .tolist() and Python control flow over tensor contents. That is fine on a
GPU but cannot be captured in a static computation graph, so the model cannot be traced,
exported to ONNX, or compiled for an NPU.

At inference the routing is top-1 and `multiply_by_gates=False`, which makes an equivalent
static formulation available:

  * MoEGate takes softmax over the top-1 logit alone, so the gate vector is exactly one-hot
    with value 1.0 at the selected expert.
  * SparseDispatcher.combine() applies exp(), index-adds the single contribution, then
    log(). For one contributing expert this composition is the identity.

Therefore  sum_i onehot_i * E_i(Z)  reproduces the dynamic path exactly, while evaluating
all M experts unconditionally. The extra cost is negligible: the experts operate on a
rank-r map (r=2) and together add well under 0.1% of model FLOPs.
"""
import math
import torch
import torch.nn.functional as F

from ..adaptation_layers import ConvLoRALinear, MoEGate


def _static_gate_forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
    """Inference-only gate: deterministic top-1, no noise, no load-balancing statistics."""
    b = feats.shape[0]
    feats_s = self.gap(feats).view(b, -1)
    logits = feats_s.float() @ self.w_gate.float()
    # Build the one-hot with broadcast equality against arange, rather than scatter or
    # one_hot. Both of those are rejected downstream: ScatterElements is emitted at opset 18
    # (the QNN converter accepts 11/13/16) and the HTP backend has no validated datatype
    # combination for OneHot. Equality against arange needs only ArgMax/Equal/Cast, which are
    # universally supported. With top-1 the softmax over a single logit is exactly 1.0, so
    # this is an exact substitute for the original gate.
    idx = logits.argmax(dim=1, keepdim=True)                                  # (B, 1)
    ar = torch.arange(self.M, device=logits.device).unsqueeze(0)              # (1, M)
    gates = (ar == idx).to(logits.dtype)                                      # (B, M) one-hot
    return gates, torch.zeros((), device=feats.device, dtype=torch.float32)


def _static_convlora_forward(self, x: torch.Tensor):
    result = F.linear(x, self.T(self.weight), bias=self.bias)
    moe_loss = torch.zeros((), device=x.device, dtype=torch.float32)
    if self.r > 0:
        lora_res = self.lora_dropout(x) @ self.lora_A.T
        dim = lora_res.dim()
        if dim == 3:
            B, L, C = lora_res.size()
            H = W = int(math.sqrt(L))
            lora_res = lora_res.reshape(B, H, W, C)
        else:
            H, W = lora_res.size()[1:3]
        lora_res = lora_res.permute(0, 3, 1, 2).contiguous()

        gates, moe_loss = self.lora_moe_gating(lora_res)

        outs = []
        for i in range(self.num_experts):
            ratio = self.upsample_ratios[i]
            cur = lora_res
            if ratio != 1:
                cur = F.interpolate(cur, scale_factor=float(ratio), mode="bicubic", align_corners=False)
            cur = self.lora_moe_experts[i](cur)
            if ratio != 1:
                cur = F.interpolate(cur, size=(int(H), int(W)), mode="bicubic", align_corners=False)
            outs.append(cur)

        stacked = torch.stack(outs, dim=1)                                   # (B, M, r, H, W)
        w = gates.view(gates.size(0), self.num_experts, 1, 1, 1).to(stacked.dtype)
        lora_res = lora_res + (stacked * w).sum(1)                           # one-hot select

        lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
        if dim == 3:
            lora_res = lora_res.reshape(B, L, C)
        result = result + (lora_res @ self.lora_B.T) * self.scaling
    return result, moe_loss


def make_static(model):
    """Re-bind every Conv-LoRA / MoEGate forward in `model` to its static equivalent."""
    import types
    n_lora = n_gate = 0
    for m in model.modules():
        if isinstance(m, ConvLoRALinear):
            m.forward = types.MethodType(_static_convlora_forward, m); n_lora += 1
        elif isinstance(m, MoEGate):
            m.forward = types.MethodType(_static_gate_forward, m); n_gate += 1
    return n_lora, n_gate
