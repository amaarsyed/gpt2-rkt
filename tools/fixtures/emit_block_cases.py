"""Emit a small pre-LN GPT-2 block fixture.

torch and transformers are pinned. The block is the eager GPT2Block in
fp32 eval mode, with TF32 off and without SDPA. Intermediates come from
that module. The script stops if the module is not the eager class.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from transformers import GPT2Config
from transformers.activations import NewGELUActivation
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block

SEQ = 3
D = 4
N_HEAD = 2
INNER = 4 * D


def fill(shape, salt):
    n = 1
    for size in shape:
        n *= size
    values = [((i * 17 + salt) % 100) / 50.0 - 1.0 for i in range(n)]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def encode(value):
    if not torch.is_tensor(value):
        raise TypeError(type(value))
    if value.dim() == 0:
        number = float(value)
        if number == float("-inf"):
            return "-inf"
        return number
    return [encode(item) for item in value]


def load_block(config, salt):
    block = GPT2Block(config)
    block.eval()
    if type(block.attn) is not GPT2Attention:
        raise SystemExit(f"expected eager attention, got {type(block.attn).__name__}")
    tensors = {
        "ln1-gain": fill((D,), salt),
        "ln1-bias": fill((D,), salt + 1),
        "c-attn-weight": fill((D, 3 * D), salt + 2),
        "c-attn-bias": fill((3 * D,), salt + 3),
        "attn-proj-weight": fill((D, D), salt + 4),
        "attn-proj-bias": fill((D,), salt + 5),
        "ln2-gain": fill((D,), salt + 6),
        "ln2-bias": fill((D,), salt + 7),
        "c-fc-weight": fill((D, INNER), salt + 8),
        "c-fc-bias": fill((INNER,), salt + 9),
        "mlp-proj-weight": fill((INNER, D), salt + 10),
        "mlp-proj-bias": fill((D,), salt + 11),
    }
    with torch.no_grad():
        block.ln_1.weight.copy_(tensors["ln1-gain"])
        block.ln_1.bias.copy_(tensors["ln1-bias"])
        block.attn.c_attn.weight.copy_(tensors["c-attn-weight"])
        block.attn.c_attn.bias.copy_(tensors["c-attn-bias"])
        block.attn.c_proj.weight.copy_(tensors["attn-proj-weight"])
        block.attn.c_proj.bias.copy_(tensors["attn-proj-bias"])
        block.ln_2.weight.copy_(tensors["ln2-gain"])
        block.ln_2.bias.copy_(tensors["ln2-bias"])
        block.mlp.c_fc.weight.copy_(tensors["c-fc-weight"])
        block.mlp.c_fc.bias.copy_(tensors["c-fc-bias"])
        block.mlp.c_proj.weight.copy_(tensors["mlp-proj-weight"])
        block.mlp.c_proj.bias.copy_(tensors["mlp-proj-bias"])
    return block, tensors


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=D,
        n_layer=2,
        n_head=N_HEAD,
        n_inner=INNER,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        activation_function="gelu_new",
        scale_attn_weights=True,
        reorder_and_upcast_attn=False,
    )
    config._attn_implementation = "eager"
    gelu = NewGELUActivation()
    x = fill((SEQ, D), 1)
    hidden = x
    blocks = []
    for index, salt in enumerate((20, 40)):
        block, tensors = load_block(config, salt)
        captured = {}

        def save_fc(mod, inputs, output, box=captured):
            box["fc"] = output.detach()

        handle = block.mlp.c_fc.register_forward_hook(save_fc)
        with torch.no_grad():
            out = block(hidden.unsqueeze(0))[0].squeeze(0)
            ln1 = block.ln_1(hidden)
            attn = block.attn(ln1.unsqueeze(0))[0].squeeze(0)
            residual1 = hidden + attn
            ln2 = block.ln_2(residual1)
            mlp_hidden = gelu(captured["fc"].squeeze(0))
        handle.remove()
        gap = (out - (residual1 + block.mlp(ln2.unsqueeze(0)).squeeze(0))).abs().max().item()
        if gap > 1e-6:
            raise SystemExit(f"block {index} residual disagrees with the module by {gap}")
        record = {name: encode(value) for name, value in tensors.items()}
        record.update({
            "ln1": encode(ln1),
            "attn": encode(attn),
            "residual1": encode(residual1),
            "ln2": encode(ln2),
            "mlp-hidden": encode(mlp_hidden),
            "out": encode(out),
        })
        blocks.append(record)
        hidden = out
    ln_f_gain = fill((D,), 7)
    ln_f_bias = fill((D,), 8)
    ln_f = nn.functional.layer_norm(hidden, (D,), ln_f_gain, ln_f_bias, 1e-5)
    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mode": "fp32 eval, eager GPT2Block, gelu_new, tf32 off, no sdpa",
        "dims": {"seq": SEQ, "d-model": D, "n-head": N_HEAD, "d-head": D // N_HEAD, "inner": INNER},
        "x": encode(x),
        "blocks": blocks,
        "ln-f-gain": encode(ln_f_gain),
        "ln-f-bias": encode(ln_f_bias),
        "ln-f": encode(ln_f),
    }
    out = Path(__file__).resolve().parents[2] / "tests" / "block-cases.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"torch {torch.__version__}; transformers {transformers.__version__}")


if __name__ == "__main__":
    main()
