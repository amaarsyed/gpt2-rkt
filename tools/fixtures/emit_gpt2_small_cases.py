"""Emit a compact GPT-2 Small eager reference.

torch 2.5.1 and transformers 4.46.3, fp32, TF32 off, eager attention.
The script reads the pinned files under models/gpt2 and writes ids,
summary stats, the full last row in fp32 and float64, top
last-position logits, and a 16-token greedy continuation. It does not
write the full tensors. The float64 row is the tighter reference.
"""

import json
from pathlib import Path

import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "gpt2"
OUT = ROOT / "tests" / "gpt2-small-cases.json"
PROMPTS = (
    "Once upon a time",
    "Hello, world",
    "The capital of France is",
    "1 + 1 =",
)
MAX_NEW = 16
TOP = 8


def last_row(tensor):
    row = tensor.detach().to(dtype=torch.float64).reshape(-1, tensor.shape[-1])[-1].cpu()
    return [float(v) for v in row]


def summarize(tensor, reference):
    flat = tensor.detach().to(dtype=torch.float64).reshape(-1).cpu()
    return {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "last_row": last_row(tensor),
        "last_row_f64": last_row(reference),
    }


def forward_stages(model, ids):
    row = torch.tensor([ids])
    captured = {}

    def save_block(index):
        def hook(module, inputs, output, index=index):
            captured[index] = output[0].detach()

        return hook

    handles = [
        model.transformer.h[index].register_forward_hook(save_block(index))
        for index in (0, 5, 11)
    ]
    out = model(row, output_hidden_states=True)
    for handle in handles:
        handle.remove()
    hidden = out.hidden_states
    if len(hidden) != model.config.n_layer + 1:
        raise SystemExit(f"expected the embedding and ln_f around the blocks; got {len(hidden)}")
    if not torch.equal(captured[0], hidden[1]):
        raise SystemExit("block 0 hook does not match hidden_states[1]")
    if not torch.equal(captured[5], hidden[6]):
        raise SystemExit("block 5 hook does not match hidden_states[6]")
    return {
        "embed": hidden[0][0],
        "block0": captured[0][0],
        "block5": captured[5][0],
        "block11": captured[11][0],
        "ln_f": hidden[-1][0],
        "logits": out.logits[0, -1],
    }


def main() -> None:
    if torch.__version__.split("+")[0] != "2.5.1":
        raise SystemExit(f"expected torch 2.5.1, got {torch.__version__}")
    if transformers.__version__ != "4.46.3":
        raise SystemExit(f"expected transformers 4.46.3, got {transformers.__version__}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, attn_implementation="eager")
    model.eval()
    model64 = GPT2LMHeadModel.from_pretrained(MODEL_DIR, attn_implementation="eager").double()
    model64.eval()
    for block in model.transformer.h:
        if type(block.attn) is not GPT2Attention:
            raise SystemExit(f"expected eager attention, got {type(block.attn).__name__}")
    cases = []
    with torch.no_grad():
        for text in PROMPTS:
            ids = tokenizer.encode(text)
            row = torch.tensor([ids])
            fp32 = forward_stages(model, ids)
            f64 = forward_stages(model64, ids)
            logits = fp32["logits"]
            top_v, top_i = torch.topk(logits, TOP)
            logits64 = f64["logits"].detach().to(dtype=torch.float64).cpu()
            steps = []
            running = list(ids)
            for _ in range(MAX_NEW):
                step_logits = model(torch.tensor([running])).logits[0, -1]
                nxt = int(torch.argmax(step_logits).item())
                steps.append(nxt)
                running.append(nxt)
                if nxt == model.config.eos_token_id:
                    break
            generated = model.generate(
                row,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                pad_token_id=model.config.eos_token_id,
            )[0, len(ids) :].tolist()
            if generated != steps:
                raise SystemExit(f"generate disagreed with the step loop for {text!r}")
            visible = [token for token in generated if token != model.config.eos_token_id]
            cases.append(
                {
                    "text": text,
                    "ids": ids,
                    "greedy_ids": generated,
                    "greedy_text": tokenizer.decode(visible),
                    "top_logits": [
                        {"i": int(i), "v": float(v)} for i, v in zip(top_i.tolist(), top_v.tolist())
                    ],
                    "top_logits_f64": [
                        {"i": int(i), "v": float(logits64[i])} for i in top_i.tolist()
                    ],
                    "stages": {
                        name: summarize(fp32[name], f64[name])
                        for name in ("embed", "block0", "block5", "block11", "ln_f")
                    },
                }
            )
    payload = {
        "torch": "2.5.1",
        "transformers": "4.46.3",
        "attention": "eager",
        "dtype": "fp32",
        "float64_reference": "last_row_f64",
        "max_new_tokens": MAX_NEW,
        "prompts": cases,
    }
    OUT.write_text(json.dumps(payload) + "\n")
    print(f"wrote {OUT}")
    for case in cases:
        print(case["text"], case["ids"], case["greedy_ids"], repr(case["greedy_text"]))


if __name__ == "__main__":
    main()
