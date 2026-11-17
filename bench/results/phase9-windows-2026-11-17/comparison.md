# phase9 speed comparison

Candidate: `8ed5dc619ae67d3dbf9a4a691f7c0727eae7fcd7`. Baseline: `fda38fb`.

Alternating trees, one warmup per child, median of 3 fresh-process samples.
Generation includes prompt prefill. Allocated bytes count only the timed call.
Process peaks include loading and are not OS working-set measurements.

| Configuration | Baseline ms | phase9 ms | Speedup |
| --- | ---: | ---: | ---: |
| load 0 | 3241.730 | 3091.165 | 1.05 |
| prefill 16 | 234.159 | 232.087 | 1.01 |
| prefill 128 | 1013.341 | 986.489 | 1.03 |
| prefill 512 | 4563.970 | 4573.892 | 1.00 |
| prefill 1000 | 11527.865 | 11462.071 | 1.01 |
| decode 128 | 4596.740 | 4567.006 | 1.01 |
| sample 128 | 4705.441 | 4698.893 | 1.00 |
| head 1 | 7.467 | 7.495 | 1.00 |
| square 768 | 29.767 | 30.702 | 0.97 |
