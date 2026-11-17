"""Emit text to token-id cases from tiktoken's gpt2 encoding.

The BPE files are downloaded from the pinned Hugging Face revision and
checked before the cases are written. tiktoken is the independent
reference; this script does not import the Racket tokenizer.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import tiktoken

REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
BASE = f"https://huggingface.co/openai-community/gpt2/resolve/{REVISION}"
FILES = {
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
}

CASES = [
    ("empty", ""),
    ("hello", "hello"),
    ("hello-leading-space", " Hello"),
    ("hello-trailing-space", "hello "),
    ("repeated-spaces", "hello  world"),
    ("leading-repeated-space", "  a"),
    ("trailing-repeated-spaces", "a  b  "),
    ("two-spaces", "  "),
    ("newline", "\n"),
    ("double-newline", "\n\n"),
    ("tab", "\t"),
    ("crlf", "\r\n"),
    ("nbsp", "\u00A0"),
    ("nbsp-word", "\u00A0hello"),
    ("newline-nbsp", "\n\u00A0"),
    ("ideographic-space", "\u3000"),
    ("ideographic-word", "\u3000hello"),
    ("accented", "café"),
    ("cjk", "你好"),
    ("emoji-wave", "👋"),
    ("emoji-smile", "🙂"),
    ("contraction-lower", "I'm"),
    ("contraction-upper", "I'M"),
    ("contraction-dont", "don't"),
    ("contraction-dont-upper", "DON'T"),
    ("apostrophe-s", "'s"),
    ("apostrophe-s-upper", "'S"),
    ("digits", "12345"),
    ("punctuation", "..."),
    ("hyphenated", "GPT-2"),
    ("sentence", "The quick brown fox"),
    ("endoftext", "<|endoftext|>"),
    ("endoftext-between", "a<|endoftext|>b"),
    ("endoftext-padded", " <|endoftext|> "),
    ("not-the-special-token", "endoftext"),
]


def download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = dest / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        url = f"{BASE}/{name}"
        data = urllib.request.urlopen(url, timeout=60).read()
        got = hashlib.sha256(data).hexdigest()
        if got != expected:
            raise SystemExit(f"sha256 mismatch for {name}: {got}")
        path.write_bytes(data)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    download(root / "models" / "gpt2")
    enc = tiktoken.get_encoding("gpt2")
    cases = []
    for name, text in CASES:
        ids = enc.encode(text, allowed_special={"<|endoftext|>"})
        if enc.decode(ids) != text:
            raise SystemExit(f"reference did not round-trip {name}")
        cases.append({"name": name, "text": text, "ids": ids})
    out = root / "tests" / "tokenizer-cases.json"
    payload = {
        "reference": "tiktoken gpt2",
        "revision": REVISION,
        "cases": cases,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
