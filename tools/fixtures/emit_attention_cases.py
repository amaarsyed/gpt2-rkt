"""Emit one small causal-attention fixture.

torch and transformers are pinned. The layer runs in fp32 eval mode.
TF32 is off, and the module is the eager GPT2Attention, not SDPA.
Scores in the fixture use -inf above the diagonal. The script also runs
GPT2Attention, whose mask is the dtype minimum, and stops if that output
disagrees with the -inf scores.
"""

import json
from pathlib import Path

import torch
import transformers
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

SEQ = 4
D = 8
N_HEAD = 2
D_HEAD = D // N_HEAD


def fill(shape, salt):
    n = 1
    for size in shape:
        n *= size
    values = [((i * 17 + salt) % 100) / 50.0 - 1.0 for i in range(n)]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def encode(value):
    if value.dim() == 0:
        number = float(value)
        if number == float("-inf"):
            return "-inf"
        return number
    return [encode(item) for item in value]


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)

    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=D,
        n_layer=1,
        n_head=N_HEAD,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        scale_attn_weights=True,
        reorder_and_upcast_attn=False,
    )
    attn = GPT2Attention(config)
    attn.eval()
    if type(attn) is not GPT2Attention:
        raise SystemExit(f"expected eager GPT2Attention, got {type(attn).__name__}")

    x = fill((SEQ, D), 1)
    weight = fill((D, 3 * D), 2)
    bias = fill((3 * D,), 3)
    proj_w = fill((D, D), 4)
    proj_b = fill((D,), 5)
    with torch.no_grad():
        attn.c_attn.weight.copy_(weight)
        attn.c_attn.bias.copy_(bias)
        attn.c_proj.weight.copy_(proj_w)
        attn.c_proj.bias.copy_(proj_b)
        mixed = x @ weight + bias
        q, k, v = mixed.split(D, dim=-1)
        qh = q.view(SEQ, N_HEAD, D_HEAD).permute(1, 0, 2)
        kh = k.view(SEQ, N_HEAD, D_HEAD).permute(1, 0, 2)
        vh = v.view(SEQ, N_HEAD, D_HEAD).permute(1, 0, 2)
        scale = D_HEAD ** -0.5
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        upper = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(upper, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, vh)
        merged = ctx.permute(1, 0, 2).contiguous().view(SEQ, D)
        output = merged @ proj_w + proj_b
        module_out = attn(x.unsqueeze(0), use_cache=False)[0].squeeze(0)

    gap = (output - module_out).abs().max().item()
    if gap > 1e-5:
        raise SystemExit(f"eager GPT2Attention disagrees with -inf scores by {gap}")

    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mode": "fp32 eval, eager GPT2Attention, tf32 off, no sdpa",
        "dims": {"seq": SEQ, "d-model": D, "n-head": N_HEAD, "d-head": D_HEAD},
        "x": encode(x),
        "c-attn-weight": encode(weight),
        "c-attn-bias": encode(bias),
        "c-proj-weight": encode(proj_w),
        "c-proj-bias": encode(proj_b),
        "q": encode(q),
        "k": encode(k),
        "v": encode(v),
        "scores": encode(scores),
        "probs": encode(probs),
        "output": encode(output),
        "eager-output-max-abs-gap": gap,
    }
    out = Path(__file__).resolve().parents[2] / "tests" / "attention-cases.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"torch {torch.__version__}; transformers {transformers.__version__}; eager gap {gap}")


if __name__ == "__main__":
    main()
