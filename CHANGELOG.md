# Changelog

## Unreleased

- Read packed query rows and write key/value projections straight to cache.
- Add optional NN matrix-worker budgets, leaving the host-count default intact.
- Record phase 9 head experiments, worker screening, and baseline comparison.

- Use one future block per processor for matrix products.
- Compute only visible causal attention, with request-owned score storage.
- Size generation caches to the prompt and requested output length.
- Reuse fresh product buffers for GELU and residual additions.
- Select top-k IDs with a bounded heap, preserving lower IDs on ties.
- Add alternating baseline/phase 8 measurements and timed allocation counts.

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
