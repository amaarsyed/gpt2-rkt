# Bug log

Defects found while building gpt2-rkt. Add one row per bug.

| Symptom | Cause | Fix | Commit | Regression check |
| --- | --- | --- | --- | --- |
| `generate --prompt -h` printed help instead of storing the prompt | `parse-generate-args` treated `-h` and `--help` as help anywhere in the argument list, including option values | Help flags count only in flag position. The following argument is the value | `95b5ea2` | `tests/cli.rkt`: "help flags in value position are values" |
| A value beginning with `--` was rejected as a missing flag value | `take-option` refused the next argument when it matched `^--` | The next argument is the value, including text that starts with `-` | `95b5ea2` | `tests/cli.rkt`: "help flags in value position are values" |
| A softmax row that contains `+inf.0` is all NaN | The max-subtract step computes `+inf.0 - +inf.0`, which is NaN, and `exp` of that stays NaN | Positions equal to `+inf.0` share the row equally and every other position is 0 | `6e4b4a5` | `tests/nn.rkt`: "softmax stays finite when a row contains +inf" |
| `"  a"` was split as two spaces and then `a`, so the leading space was not part of the next word | The trailing-whitespace alternative used `(?!\s)` ("not followed by whitespace") instead of `(?!\S)` ("not followed by a non-whitespace") | The alternative is `whitespace+(?![^whitespace])`, which backtracks until the next character is whitespace or the end | `a6af301` | `tests/pretokenize.rkt`: "trailing and repeated spaces split like GPT-2" |
| `src/embed.rkt` did not compile: `flvector-set!` was unbound | `racket/base` does not provide the flvector operations | Require `racket/flonum` | `f296560` | `tests/embed.rkt` |
| The reference-id test did not compile: `system*` and `file->bytes` were unbound | `system*` is in `racket/system` and `file->bytes` is in `racket/file` | Require both modules | `21be015` | `tests/tokenizer-cases.rkt` |
| The source guard would have failed on the pre-tokenizer comment | The comment named the reference language | Describe the whitespace class without that name | `3199901` | CI grep over `*.rkt` files |
| The BPE fetch script fails on a machine without `sha256sum` | The script always invoked `sha256sum`, which macOS does not ship | Use `sha256sum` when it exists, otherwise `shasum -a 256`. Digests come from `tools/gpt2-assets.sha256` | `15421c3` | `sh tools/fetch-gpt2-vocab.sh` |
| The tokenizer fixture test spawns `sha256sum` | The digest check shelled out with `system*` | Hash the files with `sha256-bytes` and `bytes->hex-string`, reading the same digest file | `3077ca4` | `tests/tokenizer-cases.rkt`: "pinned BPE files match the published digests" |
| A blank line in `tools/gpt2-assets.sha256` makes the digest test error | `read-asset-digests` required every line to hold a digest and a name | Skip blank lines, as the fixture loader does | `7415d6e` | `tests/tokenizer-cases.rkt`: "digest lines may be blank" |
| The attention fixture test did not compile: `list->flvector` was unbound | `racket/flonum` provides `flvector` and `make-flvector`, not `list->flvector` | Fill a `flvector` with `make-flvector` and `flvector-set!` | `8683756` | `tests/attention.rkt` |
| The attention fixture script could not import `GPT2Attention` | transformers 4.46.3 does not export that class from the package root | Import it from `transformers.models.gpt2.modeling_gpt2` | `8683756` | `tools/fixtures/emit_attention_cases.py` |

## Anticipated risks verified

Risks called out before implementation. The code already matched the intended behavior, and the test locks it.

| Risk | Result | Test |
| --- | --- | --- |
| A matmul shape mismatch must name the operand shapes | The contract error includes both dimension lists | `tests/nn.rkt`: "matmul reports shapes when the contraction does not match" |
| LayerNorm of a constant row is the bias (biased variance is 0, epsilon does not leak into the output) | Output equals the bias exactly | `tests/nn.rkt`: "layer-norm of a constant row is the bias" |
| Byte map must round-trip all 256 bytes, with space at U+0120 | Every byte maps to a distinct character and back | `tests/bpe-bytes.rkt`: "byte map round-trips every byte" |
| Contractions are case-sensitive (`'m` merges, `'M` does not) | Lowercase patterns only | `tests/pretokenize.rkt`: "contractions are case-sensitive" |
| ASCII-only `\s` splits a newline followed by NBSP | That pair stays one piece. NBSP and U+3000 are in the whitespace class | `tests/pretokenize.rkt`: "Unicode whitespace is not treated as punctuation" |
| `merges.txt` starts with `#version: 0.2`, which must not take rank 0 | The line is skipped, and the first pair keeps rank 0 | `tests/merges.rkt`: "the #version line is not a merge" |
| U+001C–U+001F might be whitespace for one reference tokenizer and punctuation for the other | tiktoken 0.9.0 and the slow GPT2Tokenizer from transformers 4.46.3 encode the bare characters and a leading space the same way, so the fixture keeps those ids. `"\n\n\x1ca"` is `[198, 198, 216, 64]`; treating U+001C as whitespace would merge the newlines into `628` | `tests/tokenizer-cases.rkt`: "encode matches the reference ids and decode round-trips" (`file-separator`, `space-file-separator`, `newlines-file-separator`, and the group, record, and unit separators) |
| Head blocks might be interleaved instead of contiguous columns | Head 0 is columns `[0, d-head)` and the next head starts at `d-head`. A dot that lives in column 1 stays in head 0 | `tests/attention.rkt`: "a head is a contiguous column block" |
| `q`, `k`, and `v` might be split on the wrong axis | `c_attn`'s last axis is three contiguous thirds, `q` then `k` then `v` | `tests/attention.rkt`: "q, k, and v are contiguous thirds of the last axis" |
| The causal mask might be `+inf` above the diagonal, or might hide the diagonal | Keys with `j > i` are `-inf.0`. The diagonal and the lower triangle stay finite, and a masked probability is 0 | `tests/attention.rkt`: "the causal mask is -inf strictly above the diagonal" |
| The score scale might be `1/sqrt(64)` for every head width | The scale is `1/sqrt(d-head)`. A head of 64 is exactly 0.125, and a head of 4 is 0.5 | `tests/attention.rkt`: "attention scale is 1 over the square root of the head width" |
