# Bug log

Defects found while building gpt2-rkt. Add one row per bug.

| Symptom | Cause | Fix | Commit | Regression check |
| --- | --- | --- | --- | --- |
| `generate --prompt -h` printed help instead of storing the prompt | `parse-generate-args` treated `-h` and `--help` as help anywhere in the argument list, including option values | Help flags count only in flag position. The following argument is the value | `95b5ea2` | `tests/cli.rkt`: "help flags in value position are values" |
| A value beginning with `--` was rejected as a missing flag value | `take-option` refused the next argument when it matched `^--` | The next argument is the value, including text that starts with `-` | `95b5ea2` | `tests/cli.rkt`: "help flags in value position are values" |

## Anticipated risks verified

Risks called out before implementation. The code already matched the intended behavior, and the test locks it.

| Risk | Result | Test |
| --- | --- | --- |
| A matmul shape mismatch must name the operand shapes | The contract error includes both dimension lists | `tests/nn.rkt`: "matmul reports shapes when the contraction does not match" |
| LayerNorm of a constant row is the bias (biased variance is 0, epsilon does not leak into the output) | Output equals the bias exactly | `tests/nn.rkt`: "layer-norm of a constant row is the bias" |
