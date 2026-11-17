# gpt2-rkt

The GPT-2 Small inference engine is written entirely in Racket, with no BLAS and no FFI.

GPT-2 Small (124M) is small enough to read and large enough to talk. This repository is that model in Racket: the byte-level tokenizer, the safetensors loader, the transformer, and the sampler. Arithmetic is Racket flonums. Parallelism is Racket futures on Racket CS. There is no C extension and no Python at runtime. The weights are the usual Hugging Face file. The forward pass is checked against an eager PyTorch reference.

Python scripts generate reference fixtures, PowerShell scripts coordinate Windows benchmarks, and shell scripts fetch assets and launch benchmarks. These are supporting tools; tokenization, weight loading, model computation, and generation run entirely in Racket. The engine is pure Racket; the repository includes tooling in other languages.

## Quickstart

Install the full [Racket CS 9.3](https://racket-lang.org/download/) distribution so `racket` and `raco` are on your PATH. The download page covers Windows, macOS, and Linux. The minimal install does not include `raco test`. This tree is tested on Racket CS 9.3.

The commands below are POSIX shell. They run on Linux and macOS. On Windows, run them from Git Bash, WSL, or another environment that provides `sh` and `curl`. Once `racket` is on your PATH, `racket main.rkt` itself is the same on all three systems.

From the repository root:

```sh
sh tools/fetch-gpt2-vocab.sh
racket main.rkt generate \
  --model-dir models/gpt2 \
  --prompt "Once upon a time" \
  --max-new-tokens 16
```

The fetch script downloads the pinned GPT-2 Small files and checks their digests. The weights are not in git. `GPT2_FETCH_WEIGHTS=0` fetches only the tokenizer and config.

Generation writes text to stdout and load time, memory, and tokens per second to stderr. Leave the sampling flags off for greedy decoding. To sample:

```sh
--temperature 0.8 --top-k 40 --seed 1
```

Temperature has to be a finite number greater than 0. Top-k and seed require a temperature. The seed is an integer from 0 through 2147483647. The prompt plus the new tokens must fit in `min(1024, n_positions)`. Generation stops at the end-of-text token or at the requested count. `racket main.rkt generate --help` lists the flags. A flag value is the next argument, including text that starts with `-`.

From Racket, the three `#f` arguments omit temperature, top-k, and seed, which is greedy:

```racket
#lang racket/base
(require "main.rkt")

(define result
  (generate (generate-config "models/gpt2" "Once upon a time" 16 #f #f #f)))
(displayln (generation-text result))
```

The program prints the new text, not the prompt again. For that prompt, the tests expect this greedy continuation of 16 tokens:

```
, the world was a place of great beauty and great danger. The world was
```

That string is the checked reference in `tests/gpt2-small-cases.json`, not a fresh sample from this machine.

## How it works

The prompt is encoded with GPT-2's byte-level BPE. Token and position embeddings are added, then twelve pre-norm blocks, a final layer norm, and a tied output head against the token table. The first call fills a key/value cache sized to the request. Each later token is one row. Sampling is greedy, or temperature with an optional top-k.

The walkthrough, including the futures split and the memory choices, is in [Architecture](docs/ARCHITECTURE.md).

## Performance

These are medians. They are not a single score, and neither one is an Apple Silicon run. There is no M5 Pro measurement in the repository.

On a Windows laptop (AMD Ryzen 5 8645HS, 12 logical processors, Racket CS 9.3), the current sources, three samples:

| Workload | Median |
| --- | ---: |
| Load | 3091.165 ms |
| Prefill, 128 tokens | 986.489 ms |
| Prefill, 1000 tokens | 11462.071 ms |
| Greedy, 128 new tokens | 28.03 tok/s |
| Sampled, 128 new tokens | 27.24 tok/s |

Greedy and sampled times include the four-token prompt. Sampling used temperature 0.8, top-k 40, and seed 1. The 28.03 tok/s figure is 128 tokens in 4567.006 ms.

On a cloud VM (Intel Xeon, 4 logical processors, Racket CS 9.3), an earlier revision of this engine did the same 128-token greedy run at 9.3366 tok/s. Load took 5.0367 s. After a collection, the process held 1,085,651,056 bytes. That run is not the current sources.

The full tables, the earlier laptop comparisons, and what the counters mean are in [Benchmarks](docs/BENCHMARKS.md). Record your own machine with `sh bench/run.sh`. Allow five minutes or more, and leave the machine idle.

## Testing

`raco test .` runs the suite. The tokenizer, merges, and config have to be fetched first. The real-weight comparison in `tests/gpt2-small.rkt` also needs `model.safetensors`, and it skips itself when that file is absent. It checks hidden states, logits, and the greedy tokens against an eager fp32 fixture, and it checks the full last row against a float64 copy.

CI does both: one job fetches only the tokenizer, and another downloads the weights and runs the eager comparison. Python under `tools/fixtures/` produced the fixtures. The engine and the tests do not call it.

## Limitations

Arithmetic is Racket flonums. There is no BLAS and no GPU. Decoding is greedy, or temperature with an optional top-k. There is no beam search. The context is `min(1024, n_positions)`. The engine does not train, and dropout is off. A leftover partial UTF-8 sequence at the end of generation becomes one U+FFFD per unfinished byte.

More detail is in the [0.1.0 notes](docs/RELEASE-v0.1.0.md). The [changelog](CHANGELOG.md) is the same version. [Speed notes](docs/dev/speed.md) record what was tried and dropped. [Bugs](docs/BUGLOG.md) records the ones that had a wrong answer or a crash.

## License

[MIT](LICENSE). Copyright 2026 Amaar.
