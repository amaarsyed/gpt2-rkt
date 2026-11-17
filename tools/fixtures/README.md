# Fixtures

Python in this directory generates independent test fixtures. It is not part of the inference engine.

`emit_tokenizer_cases.py` reads digests from `tools/gpt2-assets.sha256`, checks the pinned BPE files, and writes `tests/tokenizer-cases.json`. Each case is encoded with tiktoken 0.9.0 and with transformers 4.46.3's slow `GPT2Tokenizer`. tiktoken uses its own gpt2 BPE. The slow tokenizer reads the pinned vocab and merges. The script stops if those two id lists differ. Racket code and RackUnit tests must not spawn Python or import this directory. Racket CI does not install Python and does not run anything here.
