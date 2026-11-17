"""Emit a tiny eager GPT-2 and its greedy continuation.

torch 2.5.1 and transformers 4.46.3, fp32, TF32 off, gelu_new, dropout 0.
The script stops if attention is not the eager GPT2Attention class.
Weights are little-endian F32 safetensors, the file the CLI already requires.
"""

import json
from pathlib import Path

import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

OUT = Path(__file__).resolve().parents[2] / "tests" / "tiny-gpt2"
D = 16
N_HEAD = 2
N_LAYER = 2
N_POS = 32
INNER = 64
VOCAB = 257
EOS = 256
PROMPT = "hi"
MAX_NEW = 4


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    cs = list(bs)
    n = 0
    present = set(bs)
    for b in range(256):
        if b not in present:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def fill(shape, salt):
    n = 1
    for size in shape:
        n *= size
    # 10007 is prime and larger than vocab * width (257 * 16), so a row
    # is not a copy of a row 25 places earlier.
    values = [((i * 17 + salt * 3) % 10007) / 5003.0 - 1.0 for i in range(n)]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def write_safetensors(path, tensors):
    header = {}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        raw = tensor.detach().contiguous().numpy().astype("<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((8 - (len(payload) % 8)) % 8)
    with path.open("wb") as out:
        out.write(len(payload).to_bytes(8, "little"))
        out.write(payload)
        for blob in blobs:
            out.write(blob)


def main() -> None:
    if torch.__version__.split("+")[0] != "2.5.1":
        raise SystemExit(f"expected torch 2.5.1, got {torch.__version__}")
    if transformers.__version__ != "4.46.3":
        raise SystemExit(f"expected transformers 4.46.3, got {transformers.__version__}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=N_POS,
        n_embd=D,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_inner=INNER,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        activation_function="gelu_new",
        scale_attn_weights=True,
        reorder_and_upcast_attn=False,
        eos_token_id=EOS,
        bos_token_id=EOS,
    )
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config)
    model.eval()
    for block in model.transformer.h:
        if type(block.attn) is not GPT2Attention:
            raise SystemExit(f"expected eager attention, got {type(block.attn).__name__}")
    with torch.no_grad():
        model.transformer.wte.weight.copy_(fill((VOCAB, D), 1))
        # A zero row keeps the tied eos logit at 0, so this short greedy
        # run produces real tokens. The stop itself is covered separately.
        model.transformer.wte.weight[EOS].zero_()
        model.transformer.wpe.weight.copy_(fill((N_POS, D), 2))
        model.transformer.ln_f.weight.copy_(fill((D,), 3))
        model.transformer.ln_f.bias.copy_(fill((D,), 4))
        for index, block in enumerate(model.transformer.h):
            salt = 20 + index * 20
            block.ln_1.weight.copy_(fill((D,), salt))
            block.ln_1.bias.copy_(fill((D,), salt + 1))
            block.attn.c_attn.weight.copy_(fill((D, 3 * D), salt + 2))
            block.attn.c_attn.bias.copy_(fill((3 * D,), salt + 3))
            block.attn.c_proj.weight.copy_(fill((D, D), salt + 4))
            block.attn.c_proj.bias.copy_(fill((D,), salt + 5))
            block.ln_2.weight.copy_(fill((D,), salt + 6))
            block.ln_2.bias.copy_(fill((D,), salt + 7))
            block.mlp.c_fc.weight.copy_(fill((D, INNER), salt + 8))
            block.mlp.c_fc.bias.copy_(fill((INNER,), salt + 9))
            block.mlp.c_proj.weight.copy_(fill((INNER, D), salt + 10))
            block.mlp.c_proj.bias.copy_(fill((D,), salt + 11))
        model.lm_head.weight = model.transformer.wte.weight

    chars = bytes_to_unicode()
    vocab = {chars[b]: b for b in range(256)}
    vocab["<|endoftext|>"] = EOS
    prompt_ids = [vocab[chars[ord(ch)]] for ch in PROMPT]
    if prompt_ids != [104, 105]:
        raise SystemExit(f"prompt ids drifted: {prompt_ids}")

    steps = []
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(MAX_NEW):
            logits = model(torch.tensor([ids])).logits[0, -1]
            nxt = int(torch.argmax(logits).item())
            steps.append({"context": list(ids), "next": nxt, "logits": [float(v) for v in logits]})
            ids.append(nxt)
            if nxt == EOS:
                break
        generated = model.generate(
            torch.tensor([prompt_ids]),
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=EOS,
        )[0, len(prompt_ids) :].tolist()
    if generated != [step["next"] for step in steps]:
        raise SystemExit(f"generate disagreed with the step loop: {generated}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (OUT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False) + "\n")
    (OUT / "merges.txt").write_text("#version: 0.2\n")
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name != "lm_head.weight"
    }
    write_safetensors(OUT / "model.safetensors", state)
    reference = {
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "max_new_tokens": MAX_NEW,
        "steps": steps,
        "generated_ids": generated,
        "torch": "2.5.1",
        "transformers": "4.46.3",
        "attention": "eager",
        "dtype": "fp32",
    }
    (OUT / "reference.json").write_text(json.dumps(reference) + "\n")
    worst = max(max(abs(v) for v in step["logits"]) for step in steps)
    print(f"wrote {OUT} steps={len(steps)} ids={generated} max_abs_logit={worst}")


if __name__ == "__main__":
    main()
