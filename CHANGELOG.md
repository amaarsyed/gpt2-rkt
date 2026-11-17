# Changelog

## 0.1.0

Pure-Racket GPT-2 Small inference on Racket CS. Published numbers for a
machine come from `sh bench/run.sh` on that machine.

### Added

- GPT-2 byte-level BPE, embeddings, causal attention, a pre-LN block, and greedy, temperature, and top-k generation.
- A safetensors loader for the pinned GPT-2 Small weights, including bare Hugging Face names, an unaligned header, and a tied `lm_head.weight`.
- `sh bench/run.sh`, which checks the pinned files and writes load, prefill, decode, and matmul results for the host.
- Release notes in `docs/RELEASE-v0.1.0.md`.

### Fixed

- The benchmark reads resident memory after a collection, so the weight-file buffer is not part of that figure. `load-model` itself does not collect.
- Attention reads each head from the key and value cache in place.
- `next-token-logits` allocates a cache for the prompt, not the whole position table.
- A cache handle cannot be advanced twice.
- The eager check compares every channel of the last row with a float64 reference, beside the fp32 plan bar.
