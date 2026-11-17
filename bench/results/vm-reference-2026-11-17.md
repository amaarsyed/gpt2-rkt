Reference run, not the M5 Pro.

label: vm-reference
date: 2026-11-17
cpu: Intel(R) Xeon(R) Processor
physical-cpu: unavailable
ncpu: unavailable
processors: 4
racket: 9.3 (chez-scheme)
git: 14746964b28ee2deb1c387745a45618207fd2fcd
prompt: Once upon a time
seed: 1
decoding: greedy, warmup 1, median of 3
note: Reference run on the cloud VM, not the M5 Pro.

Each row is its own process. Peak is current-memory-use 'peak in that process.

| configuration | median | peak bytes | pid |
| --- | --- | ---: | ---: |
| load | 5.0367 s | 2,248,894,968 | 22444 |
| prefill 16 | 488.9253 ms | 2,248,844,344 | 22445 |
| prefill 128 | 1837.0529 ms | 2,248,863,000 | 22473 |
| prefill 512 | 8394.3052 ms | 2,248,857,768 | 22481 |
| prefill 1000 | 20660.1813 ms | 2,248,871,536 | 22494 |
| decode 128 | 9.3366 tok/s | 2,248,894,184 | 22505 |
| matmul 768×768×768 | 14.6842 GFLOP/s | 116,141,152 | 22534 |
| matmul 1×768×50257 | 6.0236 GFLOP/s | 413,302,272 | 22539 |

load resident after collection: 1,085,651,056 bytes
decode produced 128 ids, first 11.
