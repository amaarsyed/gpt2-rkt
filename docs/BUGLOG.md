# Bug log

Defects found while building gpt2-rkt. Add one row per bug.

| Symptom | Cause | Fix | Regression check |
| --- | --- | --- | --- |
| `generate --prompt -h` printed help instead of storing the prompt | `parse-generate-args` treated `-h` and `--help` as help anywhere in the argument list, including option values | Help flags count only in flag position. The following argument is the value | `tests/cli.rkt`: "help flags in value position are values" |
| A value beginning with `--` was rejected as a missing flag value | `take-option` refused the next argument when it matched `^--` | The next argument is the value, including text that starts with `-` | `tests/cli.rkt`: "help flags in value position are values" |
