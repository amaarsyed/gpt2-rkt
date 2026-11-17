# Fixtures

Python in this directory generates independent test fixtures. It is not part of the inference engine.

`emit_tokenizer_cases.py` checks the pinned BPE files and writes `tests/tokenizer-cases.json` from tiktoken. Racket code and RackUnit tests must not spawn Python or import this directory. Racket CI does not install Python and does not run anything here.
