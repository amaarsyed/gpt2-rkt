# Phase 6 and phase 7 on the same machine

Date: 2026-11-17. CPU: AMD Ryzen 5 8645HS. Racket: 9.3 CS. Processors: 12.
Phase 6: `4e0631f4bb12ebd9e11182174de6ec808195568d`. Phase 7: `0f32e9b23836e07dd18c7d83c00caf84bb2c1e0b`.
Common driver: `f8e414d3a104fa0428ef1f1292b5dcb4999a19df`.

One warmup, then the median of 3 calls. Each row uses a fresh process.
Times are milliseconds. Speedup is phase 6 time / phase 7 time; above 1 is faster.
Peak bytes are Racket's process high-water mark, including model loading, not OS working set.
Load time excludes the collection used to measure resident bytes. Filesystem caches are warm.
Prefill includes one generated token. Decode includes the four-token prefill and 128 new tokens.
Both phases use their original EOS behavior; full requested counts and identical token IDs are checked.

| Configuration | Phase 6 ms | Phase 7 ms | Speedup | Phase 6 peak bytes | Phase 7 peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| load 0 | 3908.218 | 3244.442 | 1.20 | 2255334480 | 2255355848 |
| prefill 16 | 223.684 | 232.051 | 0.96 | 2255296128 | 2255340144 |
| prefill 128 | 1299.934 | 1336.835 | 0.97 | 2255323280 | 2255359520 |
| prefill 512 | 6507.507 | 6288.714 | 1.03 | 2255354080 | 2255366288 |
| prefill 1000 | 14299.251 | 15463.771 | 0.92 | 2255329488 | 2255328512 |
| decode 128 | 6355.449 | 5835.142 | 1.09 | 2255331536 | 2255368432 |
| square 768 | 31.930 | 30.538 | 1.05 | 108840160 | 109161920 |
| logits 1 | 7.088 | 7.812 | 0.91 | 401622208 | 407436896 |

The JSON files contain every timed sample and the generated token IDs.
Runs are sequential: phase 6 first, then phase 7. Thermal and background-load effects are not controlled.
These are Windows measurements, not the existing cloud VM or an M5 Pro benchmark.
