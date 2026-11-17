# Changelog

## 0.1.0

GPT-2 Small inference in pure Racket, on Racket CS. This is the version of the tree in the repository, including the cache and kernel work below. Numbers for a machine come from `sh bench/run.sh` on that machine.

### Added

- GPT-2 byte-level BPE, embeddings, causal attention, a pre-norm block, and greedy, temperature, and top-k generation.
- A safetensors loader for the pinned GPT-2 Small weights, including bare Hugging Face names, an unaligned header, and a tied `lm_head.weight`.
- `sh bench/run.sh`, which checks the pinned files and writes load, prefill, decode, and matmul results for the host.
- Release notes in `docs/RELEASE-v0.1.0.md`.

### Changed

- Matrix products use one future block per processor.
- Attention computes only the visible causal keys, with a score row owned by the request.
- Generation caches are sized to the prompt plus the requested output, and `next-token-logits` allocates only the prompt.
- Cached blocks read the packed query and write key and value rows straight into the cache.
- GELU and residual additions reuse the product buffer they just allocated.
- Top-k selects ids with a bounded heap and keeps the lower id on a tie.
- Matrix products take an optional worker budget. The default is still the host's processor count.

### Fixed

- The benchmark reads resident memory after a collection, so the weight-file buffer is not part of that figure. `load-model` itself does not collect.
- Attention reads each head from the key and value cache in place.
- A cache handle cannot be advanced twice.
- The eager check compares every channel of the last row with a float64 reference, beside the fp32 tolerance.
