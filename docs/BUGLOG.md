# Bug log

Defects found while building gpt2-rkt. Add one row per bug.

| Symptom | Cause | Fix | Commit | Regression check |
| --- | --- | --- | --- | --- |
| `generate --prompt -h` printed help instead of storing the prompt | `parse-generate-args` treated `-h` and `--help` as help anywhere in the argument list, including option values | Help flags count only in flag position. The following argument is the value | `95b5ea2` | `tests/cli.rkt`: "help flags in value position are values" |
| A value beginning with `--` was rejected as a missing flag value | `take-option` refused the next argument when it matched `^--` | The next argument is the value, including text that starts with `-` | `95b5ea2` | `tests/cli.rkt`: "help flags in value position are values" |
| A softmax row that contains `+inf.0` is all NaN | The max-subtract step computes `+inf.0 - +inf.0`, which is NaN, and `exp` of that stays NaN | Positions equal to `+inf.0` share the row equally and every other position is 0 | `6e4b4a5` | `tests/nn.rkt`: "softmax stays finite when a row contains +inf" |
| `"  a"` was split as two spaces and then `a`, so the leading space was not part of the next word | The trailing-whitespace alternative used `(?!\s)` ("not followed by whitespace") instead of `(?!\S)` ("not followed by a non-whitespace") | The alternative is `whitespace+(?![^whitespace])`, which backtracks until the next character is whitespace or the end | `a6af301` | `tests/pretokenize.rkt`: "trailing and repeated spaces split like GPT-2" |

## Anticipated risks verified

Risks called out before implementation. The code already matched the intended behavior, and the test locks it.

| Risk | Result | Test |
| --- | --- | --- |
| A matmul shape mismatch must name the operand shapes | The contract error includes both dimension lists | `tests/nn.rkt`: "matmul reports shapes when the contraction does not match" |
| LayerNorm of a constant row is the bias (biased variance is 0, epsilon does not leak into the output) | Output equals the bias exactly | `tests/nn.rkt`: "layer-norm of a constant row is the bias" |
| Byte map must round-trip all 256 bytes, with space at U+0120 | Every byte maps to a distinct character and back | `tests/bpe-bytes.rkt`: "byte map round-trips every byte" |
| Contractions are case-sensitive (`'m` merges, `'M` does not) | Lowercase patterns only | `tests/pretokenize.rkt`: "contractions are case-sensitive" |
| ASCII-only `\s` splits a newline followed by NBSP | That pair stays one piece. NBSP and U+3000 are in the whitespace class | `tests/pretokenize.rkt`: "Unicode whitespace is not treated as punctuation" |
