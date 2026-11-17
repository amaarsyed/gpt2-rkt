# Bug log

Defects that produced a wrong answer, a crash, or a hang while the engine was being matched to GPT-2. Each one has a regression check. The [speed notes](dev/speed.md) cover trials that were merely slower.

## Tokenizer

**A leading space was split off `"  a"`.** The trailing-whitespace alternative used `(?!\s)`, "not followed by whitespace," instead of `(?!\S)`, "not followed by a non-whitespace." The pattern is whitespace that backtracks until the next character is whitespace or the end. `tests/pretokenize.rkt`, "trailing and repeated spaces split like GPT-2."

**`src/embed.rkt` did not compile.** `flvector-set!` is not in `racket/base`. The module requires `racket/flonum`. `tests/embed.rkt`.

**The reference-id test did not compile.** `system*` lives in `racket/system` and `file->bytes` in `racket/file`. Both are required. `tests/tokenizer-cases.rkt`.

**The fetch script failed on a machine without `sha256sum`.** macOS does not ship it. The script uses `sha256sum` when it exists and `shasum -a 256` otherwise. Digests still come from `tools/gpt2-assets.sha256`.

**The tokenizer fixture test shelled out to `sha256sum`.** It now hashes with `sha256-bytes` and `bytes->hex-string`, reading the same digest file. `tests/tokenizer-cases.rkt`, "pinned BPE files match the published digests."

**A blank line in the digest file aborted the test.** `read-asset-digests` required every line to hold a digest and a name. Blank lines are skipped, as the fixture loader already did. `tests/tokenizer-cases.rkt`, "digest lines may be blank."

**A comment would have failed the source check.** Racket sources are not allowed to name `tools/fixtures` or Python. The pre-tokenizer comment named the reference language. It now describes the whitespace class without that. The CI grep over `*.rkt` is the check.

## Loader

**An empty safetensors shape printed a contract traceback.** The reader accepted `[]`, and tensors need at least one dimension. The error names the tensor. `tests/safetensors.rkt`, "an empty shape is a gpt2 error that names the tensor."

**A tensor whose byte length disagreed with its shape printed a stack trace.** `make-tensor` raised a contract error, which the CLI does not catch. The length is rejected first, and the message names the tensor. `tests/safetensors.rkt`.

**A safetensors round-trip raised "not a procedure" and printed the floats.** The data vector was named `values`, which shadowed the procedure used to return the hash entry. It is named `nums`. `tests/safetensors.rkt`, "F32 tensors round-trip through the header."

**`attn.masked_bias`, or a tied `lm_head.weight`, failed as an unexpected tensor.** The loader only skipped `attn.bias`. It now skips `masked_bias` too, and keeps `lm_head.weight` only when it equals `wte`. `tests/model.rkt`, "masked_bias is skipped and a tied lm_head is accepted."

**Hashing the weights loaded the whole file next to the model.** `check-file-sha256` used `file->bytes`. `sha256-bytes` reads the input port. `tests/model.rkt`, "a sha256 mismatch names the file and the fetch command."

**`load-bytes` was about 1.67 GB.** `current-memory-use` still counted the dead file buffer, about 548 MB. The benchmark child collects before reading the counter. `load-model` does not. After that, the same load is 1,085,651,056 bytes. `tests/gpt2-small.rkt` prints `load-bytes`.

**The attention fixture script could not import `GPT2Attention`.** transformers 4.46.3 does not export that class from the package root. The import is `transformers.models.gpt2.modeling_gpt2`.

**The two-block test raised a shape error on `c_attn`.** The generated weight length was the last dimension, not the product of the dimensions. `tests/block.rkt`, "two blocks then ln_f match the folded pre-LN stack."

**`n_head: 0` escaped the CLI as a division error.** Dimensions were checked for presence, not range, before `remainder`. Ranges are checked first. A model with zero blocks is still legal. `tests/model.rkt`, "invalid config dimensions are named before arithmetic" and "a model without blocks still loads."

## Attention, the block, and sampling

**`list->flvector` and `build-flvector` were unbound.** `racket/flonum` provides `flvector` and `make-flvector`. The tests fill a vector with `make-flvector` and `flvector-set!`. `tests/attention.rkt`, `tests/block.rkt`.

**A softmax row that contains `+inf.0` was all NaN.** Subtracting the row max computes `+inf.0 - +inf.0`. Positions equal to `+inf.0` share the row, and every other position is 0. `tests/nn.rkt`, "softmax stays finite when a row contains +inf."

**A new dense-attention comparison failed on a 129-row input.** The test demanded bit identity between two accumulation orders. The cross-check uses a `1e-12` bound. Ownership checks and the eager tolerances stayed exact. `tests/attention.rkt`, "causal rows match dense attention, including parallel heads and cache capacity."

**Each decode step copied the filled key and value prefix.** `slice-cols` allocated that prefix once per head for keys and again for values. At 1,024 positions that is about 151 MB of copies per token. The head reads its columns at a column offset and the cache's row stride. `tests/attention.rkt`, "the last query row matches that row of the full attend."

**`next-token-logits` allocated a 1,024-row cache to score a short prompt.** `empty-kv-cache` always used `n_positions`, about 151 MB of zeros. The capacity is the prompt length. `tests/model.rkt`, "a kv cache is sized to the call and cannot be advanced twice."

**Calling `model-forward` twice on one cache overwrote the first call's rows.** The buffers are mutated in place, and the old handle stayed usable. The handle's box is cleared when the forward starts. The returned cache is the one that continues. Same test as above.

**The eager check sampled only 8 to 32 channels.** `n // 32` shares a factor with 768, so whole head ranges were covered only by the min, the max, and the mean. The fixture stores the full last row, and a float64 copy of that row is checked at `1e-6 + 1e-8 * |e|`. `tests/gpt2-small.rkt`, "eager hidden states, logits, and greedy tokens match."

**A truncated output could pass the eager tolerance check.** Parallel list iteration stopped at the shorter list. Lengths are compared first, and an unequal length is in the error summary. `tests/fixture-util.rkt`, "a matching prefix is not a matching fixture."

**The top-k regression file did not parse.** The outer `for` was missing a closing parenthesis. `tests/model.rkt` and `tests/tiny-model.rkt`.

**A tiny legal temperature selected the last vocabulary id.** An all-negative row divided by that temperature became all `-inf.0`, and the sampler fell through. The row max is subtracted before the division. `tests/model.rkt`, "a tiny temperature keeps the largest negative logit."

**The tiny greedy match could not see a 25-row shift.** The fill repeated every 25 rows, so each step's max logit was an 11-way tie. The fill uses a prime modulus larger than the embedding table. `tests/tiny-model.rkt`, "tiny eager logits and the greedy continuation match."

**A register-kernel trial printed `#<procedure:+>` before its JSON.** A stray `+` expression was left at the top level of a prototype. The prototype was slower and was discarded. The JSON measurements were kept. See the kernel section of [Benchmarks](BENCHMARKS.md).

## The command line

**`generate --prompt -h` printed help.** `-h` and `--help` counted as help anywhere in the argument list, including option values. They count only in flag position. The following argument is the value. `tests/cli.rkt`, "help flags in value position are values."

**A value beginning with `--` was rejected as a missing flag value.** The next argument is the value, including text that starts with `-`. Same test.

**`--max-new-tokens 40` on the 32-position tiny model passed the length check, then printed a stack trace.** The check compared against the global 1,024. The later position-embed failure is a contract error the CLI does not catch. Prompt plus new tokens is compared with `min(1024, n_positions)` before generation starts. `tests/model.rkt`, "generation uses the model's position count." `tests/cli.rkt`, "a short position table rejects a long generation."

**`--seed 3000000000` escaped the CLI as a contract error.** `generate-config` accepted any nonnegative integer, and `random-seed` only accepts 0 through 2147483647. A seed outside that range is rejected before seeding. `tests/cli.rkt`, "flag validation."

## Benchmarks and the host

**A benchmark child that fills the error pipe hangs the parent.** Stdout was read to EOF before stderr, so both sides waited. The two pipes are read together. `tests/bench.rkt`, "a child that fills stderr still returns."

**The CPU-name test failed on aarch64 Linux.** `/proc/cpuinfo` has no `model name` line, and there is no sysctl brand string. The reader uses `Model` or `Hardware` when present, otherwise the architecture name, and it looks for `/usr/sbin/sysctl` when that is not on `PATH`. `tests/harness.rkt` and `tests/bench.rkt`.

**A missing `git` wrote `"git_sha": false`.** `(and git ...)` returns `#f`, which JSON prints as `false`. The missing case is the string `unknown`. `tests/harness.rkt`.

## Details the tests pin down

These are the GPT-2 choices that were easy to get backwards. The test is what locks the choice.

| Easy mistake | What the code does | Test |
| --- | --- | --- |
| A matmul shape mismatch with no dimensions in the message | The contract error includes both dimension lists | `tests/nn.rkt`: "matmul reports shapes when the contraction does not match" |
| Layer norm of a constant row leaking epsilon into the output | The output equals the bias. Biased variance is 0 | `tests/nn.rkt`: "layer-norm of a constant row is the bias" |
| The byte map dropping a byte, or mapping space somewhere other than U+0120 | Every byte maps to a distinct character and back | `tests/bpe-bytes.rkt`: "byte map round-trips every byte" |
| Contractions folded to lower case | `'m` merges and `'M` does not | `tests/pretokenize.rkt`: "contractions are case-sensitive" |
| ASCII-only whitespace splitting a newline followed by NBSP | That pair stays one piece. NBSP and U+3000 are in the whitespace class | `tests/pretokenize.rkt`: "Unicode whitespace is not treated as punctuation" |
| The `#version: 0.2` line in `merges.txt` taking rank 0 | The line is skipped. The first pair keeps rank 0 | `tests/merges.rkt`: "the #version line is not a merge" |
| U+001C through U+001F treated as whitespace | tiktoken 0.9.0 and the slow GPT-2 tokenizer from transformers 4.46.3 encode the bare characters and a leading space the same way. `"\n\n\x1ca"` is `[198, 198, 216, 64]`. Treating U+001C as whitespace would merge the newlines into `628` | `tests/tokenizer-cases.rkt`: "encode matches the reference ids and decode round-trips" |
| Heads interleaved instead of contiguous columns | Head 0 is columns `[0, d-head)`. A dot in column 1 stays in head 0 | `tests/attention.rkt`: "a head is a contiguous column block" |
| Query, key, and value split on the wrong axis | `c_attn`'s last axis is three contiguous thirds, query then key then value | `tests/attention.rkt`: "q, k, and v are contiguous thirds of the last axis" |
| The causal mask `+inf` above the diagonal, or hiding the diagonal | Keys with `j > i` are `-inf.0`. The diagonal and the lower triangle stay finite. A masked probability is 0 | `tests/attention.rkt`: "the causal mask is -inf strictly above the diagonal" |
| Every head scaled by `1/sqrt(64)` | The scale is `1/sqrt(d-head)`. A head of 64 is 0.125. A head of 4 is 0.5 | `tests/attention.rkt`: "attention scale is 1 over the square root of the head width" |
| Unbiased variance, or an epsilon other than `1e-5` | `ln_1` of `[1, 3]` matches `1/sqrt(1 + 1e-5)`. A constant row is exactly the bias | `tests/block.rkt`: "the block normalizes with biased variance and eps 1e-5 before attention" |
| GELU using the erf formula | At 1.0 the tanh approximation is `0.8411919906082768`. The erf value is rejected | `tests/block.rkt`: "GELU matches the tanh approximation and not erf" |
| Post-norm, or the sublayer replacing the stream | Zero norm gain makes both sublayers see zeros, so the block returns the input. The order is `h = x + attn(ln_1(x))`, then `h + mlp(ln_2(h))` | `tests/block.rkt`: "pre-LN adds each sublayer onto the residual stream" |
| `c_fc` or `c_proj` stored transposed | `c_fc` is `[width, inner]` and `c_proj` is `[inner, width]`. The swapped `c_fc` is a contract error. A one-hot input selects `GELU(1)` | `tests/block.rkt`: "c_fc maps model width to the inner width and c_proj maps back" |
| The cache position off by one | A cached step's logits equal the full recompute of that prefix bit for bit, and both match the eager fixture | `tests/model.rkt`: "cached logits match a full recompute at every step" |
| The tied head multiplying `wte` as `[d, vocab]` | `wte` stays `[vocab, d]`. The logits use `matmul-transposed-b`. A plain `matmul` of that table is a contract error | `tests/model.rkt`: "tied logits are h dotted with each wte row" |
| Generation dropping end of text, or continuing past it | The id is included and the loop stops. The printed text does not contain it. Token 50256 is that id when the config omits `eos_token_id` | `tests/model.rkt`: "generation stops at the eos token", "generation stops on token 50256" |
| The context limit allowing 1,025 or rejecting 1,024 | 1,024 positions succeed. 1,025, and a 1,024-token prompt plus one new token, name 1,024 and do not run. A shorter position table uses that length | `tests/model.rkt`: "exactly 1024 positions are legal and 1025 is not". `tests/embed.rkt`: "position embed is 0 .. n-1 and stops at 1024" |
| A seed that does not repeat, or top-k 1 that is not greedy | The same seed repeats the ids. A different seed does not. Top-k 1 is the argmax | `tests/model.rkt`: "a seed repeats the sample and top-k 1 is greedy" |
| Temperature 0, a negative temperature, `+inf.0`, `+nan.0`, a top-k below 1 or above the vocabulary, a non-integer token count | All of those are errors, as is a prompt plus generation past the position table | `tests/cli.rkt`: "flag validation" |
| Conv1D on the real weights stored transposed | `h.0.attn.c_attn.weight` is `[768, 2304]`. A swapped shape is named. Block 0 matches the eager hidden state | `tests/model.rkt` and `tests/gpt2-small.rkt` |
| The published file required to contain `lm_head` | There is no `lm_head`. A copy that equals `wte` is accepted and unused. A different tensor is named. Last-position logits match the eager top-k | `tests/model.rkt` and `tests/gpt2-small.rkt` |
| The safetensors header required to be a multiple of 8 | The official header length is 14,283, remainder 3, and the floats still read | `tests/safetensors.rkt`: "an unaligned header and a space-padded header both read" |
| A non-F32 dtype decoded | The dtype is named and rejected. Every tensor in the published file is F32 | `tests/safetensors.rkt`: "a non-F32 dtype is rejected" |
| A missing key surfacing as a contract error | The message names the tensor | `tests/model.rkt`: "a transposed c_attn, a missing key, and an extra key are named" |
| A wrong digest accepted | A mismatch names the file and `sh tools/fetch-gpt2-vocab.sh`. The four pinned files match | `tests/model.rkt` and `tests/gpt2-small.rkt` |
| The 1,024 limit ignoring the loaded `n_positions` | The loaded model has 1,024 positions, equal to `gpt2-context-length`. 1,025 tokens name 1,024 and do not run the forward pass | `tests/gpt2-small.rkt` |
| The `1e-3` gap at block 11 being a bug in the forward pass | A float64 reference of the same last rows matches to about `1e-13`. The gap is the fp32 reference. The fp32 tolerance stays. The float64 row is checked at `1e-6 + 1e-8 * \|e\|` | `tests/gpt2-small.rkt`: "eager hidden states, logits, and greedy tokens match" |
| The benchmark producing different token ids on each run | Greedy decoding of a fixed prompt. A timed run that disagrees with the warmup is an error | `tests/bench.rkt`: "greedy ids are the same on every call" |
| Peak memory combining several configurations in one process | Each configuration is a child process. The report rejects a repeated pid | `tests/bench.rkt`: "each configuration is a different process" |
| An empty CPU name on macOS or Linux | Linux reads `model name`, then `Model` or `Hardware`, from `/proc/cpuinfo`, and otherwise uses the architecture. macOS reads `sysctl -n machdep.cpu.brand_string`, including `/usr/sbin/sysctl` when it is off `PATH`. A blank reading is not used | `tests/harness.rkt` and `tests/bench.rkt` |
| A cache handle advanced twice, so the second call overwrites the first call's rows | The handle's box is cleared when the forward starts. The returned cache is the one that continues | `tests/model.rkt`: "a kv cache is sized to the call and cannot be advanced twice" |
