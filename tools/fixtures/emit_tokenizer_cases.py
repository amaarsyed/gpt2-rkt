"""Emit text to token-id cases from two independent GPT-2 tokenizers.

tiktoken loads its own copy of the gpt2 BPE. The slow GPT2Tokenizer
reads the pinned Hugging Face vocab and merges. Every case is encoded
with both. If they disagree, this script stops and does not write a
fixture or choose a side.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import tiktoken
import transformers
from transformers import GPT2Tokenizer

REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
BASE = f"https://huggingface.co/openai-community/gpt2/resolve/{REVISION}"
SPECIAL = "<|endoftext|>"

CASES = [
    ("empty", ""),
    ("hello", "hello"),
    ("hello-leading-space", " Hello"),
    ("hello-trailing-space", "hello "),
    ("repeated-spaces", "hello  world"),
    ("leading-repeated-space", "  a"),
    ("trailing-repeated-spaces", "a  b  "),
    ("two-spaces", "  "),
    ("three-spaces", "a   b"),
    ("four-spaces", "a    b"),
    ("space-newline-space", "a \n b"),
    ("space-then-newline", " \n"),
    ("newline-run", "x\n\n\ny"),
    ("tab-run", "\t\tfoo"),
    ("newline", "\n"),
    ("double-newline", "\n\n"),
    ("tab", "\t"),
    ("vertical-tab", "\x0b"),
    ("form-feed", "\x0c"),
    ("next-line", "\u0085"),
    ("line-separator", "\u2028"),
    ("ogham-space", "\u1680"),
    ("narrow-nbsp", "\u202f"),
    ("file-separator", "\u001c"),
    ("group-separator", "\u001d"),
    ("record-separator", "\u001e"),
    ("unit-separator", "\u001f"),
    ("space-file-separator", " \u001ca"),
    ("newlines-file-separator", "\n\n\u001ca"),
    ("space-group-separator", " \u001da"),
    ("space-record-separator", " \u001ea"),
    ("space-unit-separator", " \u001fa"),
    ("zero-width-space", "a\u200bb"),
    ("byte-order-mark", "a\ufeffb"),
    ("mongolian-vowel-separator", "a\u180eb"),
    ("crlf", "\r\n"),
    ("nbsp", "\u00a0"),
    ("nbsp-word", "\u00a0hello"),
    ("newline-nbsp", "\n\u00a0"),
    ("ideographic-space", "\u3000"),
    ("ideographic-word", "\u3000hello"),
    ("space-before-punct", " !"),
    ("spaces-before-punct", "  !!"),
    ("comma-sentence", "hello, world!"),
    ("accented", "café"),
    ("decomposed-e-acute", "e\u0301"),
    ("cjk", "你好"),
    ("emoji-wave", "👋"),
    ("emoji-smile", "🙂"),
    ("zwj-emoji", "👩‍💻"),
    ("contraction-lower", "I'm"),
    ("contraction-upper", "I'M"),
    ("contraction-dont", "don't"),
    ("contraction-dont-upper", "DON'T"),
    ("apostrophe-s", "'s"),
    ("apostrophe-s-upper", "'S"),
    ("curly-apostrophe", "it’s"),
    ("digits", "12345"),
    ("superscript-two", "²"),
    ("roman-twelve", "Ⅻ"),
    ("arabic-indic-three", "٣"),
    ("digits-inside-word", "abc123def"),
    ("year-cjk", "2024年"),
    ("punctuation", "..."),
    ("hyphenated", "GPT-2"),
    ("sentence", "The quick brown fox"),
    ("paragraph",
     "The small model reads a paragraph one piece at a time, merges the "
     "most common pairs first, and keeps going until no listed pair remains. "
     "Spaces, punctuation, and the occasional number like 42 all sit in that "
     "same pass, so a long realistic paragraph is the check that rank order "
     "still holds once the input is longer than a handful of words."),
    ("endoftext", SPECIAL),
    ("endoftext-between", "a<|endoftext|>b"),
    ("endoftext-padded", " <|endoftext|> "),
    ("not-the-special-token", "endoftext"),
]


def load_digests(path: Path) -> dict:
    digests = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        digests[name] = digest
    return digests


def download(dest: Path, digests: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, expected in digests.items():
        path = dest / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        url = f"{BASE}/{name}"
        data = urllib.request.urlopen(url, timeout=60).read()
        got = hashlib.sha256(data).hexdigest()
        if got != expected:
            raise SystemExit(f"sha256 mismatch for {name}: {got}")
        path.write_bytes(data)


def codepoints(text: str) -> str:
    return " ".join(f"U+{ord(ch):04X}" for ch in text)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    digests = load_digests(root / "tools" / "gpt2-assets.sha256")
    model_dir = root / "models" / "gpt2"
    download(model_dir, digests)
    tik = tiktoken.get_encoding("gpt2")
    hf = GPT2Tokenizer(
        vocab_file=str(model_dir / "vocab.json"),
        merges_file=str(model_dir / "merges.txt"),
    )
    hf.model_max_length = 10**9
    mismatches = []
    cases = []
    for name, text in CASES:
        tik_ids = tik.encode(text, allowed_special={SPECIAL})
        hf_ids = hf.encode(text, add_special_tokens=False)
        if tik_ids != hf_ids:
            mismatches.append((name, text, tik_ids, hf_ids))
            continue
        if tik.decode(tik_ids) != text or hf.decode(hf_ids) != text:
            raise SystemExit(f"reference did not round-trip {name}")
        cases.append({"name": name, "text": text, "ids": tik_ids})
    if mismatches:
        print("tiktoken and GPT2Tokenizer disagree; not picking a reference")
        for name, text, tik_ids, hf_ids in mismatches:
            print(f"{name} {codepoints(text)}")
            print(f"  tiktoken {tik_ids}")
            print(f"  GPT2Tokenizer {hf_ids}")
        raise SystemExit(1)
    out = root / "tests" / "tokenizer-cases.json"
    payload = {
        "reference": "tiktoken gpt2 and transformers GPT2Tokenizer",
        "tiktoken": tiktoken.__version__,
        "transformers": transformers.__version__,
        "revision": REVISION,
        "cases": cases,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")
    print(f"tiktoken {tiktoken.__version__}; transformers {transformers.__version__}")


if __name__ == "__main__":
    main()
