# Phase 8 speed comparison

Candidate: `e8af302f4ce8245bae30974f609d11c20e66fa7a`. Baseline: `1ad7035`.

Alternating trees, one warmup per child, median of 3 fresh-process samples.
Generation includes prompt prefill. Allocated bytes count only the timed call.
Process peaks include loading and are not OS working-set measurements.

| Configuration | Baseline ms | Phase 8 ms | Speedup |
| --- | ---: | ---: | ---: |
| load 0 | 3167.608 | 2998.507 | 1.06 |
| prefill 16 | 219.288 | 228.170 | 0.96 |
| prefill 128 | 1176.652 | 1027.874 | 1.14 |
| prefill 512 | 6135.564 | 5030.384 | 1.22 |
| prefill 1000 | 15183.579 | 10928.493 | 1.39 |
| decode 128 | 5838.217 | 4583.654 | 1.27 |
| sample 128 | 6824.062 | 4748.245 | 1.44 |
| head 1 | 7.771 | 7.512 | 1.03 |
| square 768 | 30.012 | 30.282 | 0.99 |
