import type { StageId } from './traceTypes';

/** Verbatim Racket from gpt2-rkt @ ae29f0007eda76012ed772be4e54b18ecbf56939. Do not edit by hand. */

export interface RacketExcerpt {
  file: string;
  fn: string;
  startLine: number;
  highlightStart: number;
  highlightEnd: number;
  callSiteLine: number | null;
  callSiteEndLine: number | null;
  code: string;
}

export const RACKET_COMMIT = "ae29f0007eda76012ed772be4e54b18ecbf56939";

export const RACKET_EXCERPTS: Record<StageId, RacketExcerpt> = {
  "bpe-merge": {
    "file": "src/tokenizer.rkt",
    "fn": "bpe-pieces",
    "startLine": 115,
    "highlightStart": 117,
    "highlightEnd": 125,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "        (define final\n          (stage bpe-merge bpe-pieces\n            (let loop ([word word])\n              (define pair (lowest-pair word ranks))\n              (cond\n                [(not pair) word]\n                [else\n                 (define merged (apply-merge word (car pair) (cdr pair)))\n                 (if (null? (cdr merged))\n                     merged\n                     (loop merged))]))))"
  },
  "input-embed": {
    "file": "src/embed.rkt",
    "fn": "input-embed",
    "startLine": 77,
    "highlightStart": 81,
    "highlightEnd": 82,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "(define (input-embed wte wpe ids #:start [start 0])\n  (define obs (current-trace-observer))\n  (if-trace obs 'input-embed\n    (stage input-embed input-embed\n      (tensor-add (token-embed wte ids)\n                  (position-embed wpe (length ids) #:start start)))\n    (lambda (y t0) (trace-activation! obs 'input-embed y t0))))"
  },
  "block-ln1": {
    "file": "src/block.rkt",
    "fn": "layer-norm",
    "startLine": 146,
    "highlightStart": 149,
    "highlightEnd": 151,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "  (define ln1\n    (if-trace obs 'block-ln1\n      (stage block-ln1 layer-norm\n        (layer-norm x\n                    (block-params-ln1-gain params)\n                    (block-params-ln1-bias params)))\n      (lambda (y t0) (trace-activation! obs 'block-ln1 y t0))))"
  },
  "block-qkv": {
    "file": "src/block.rkt",
    "fn": "matmul",
    "startLine": 153,
    "highlightStart": 156,
    "highlightEnd": 158,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "  (define mixed\n    (if-trace obs 'block-qkv\n      (stage block-qkv matmul\n        (matmul ln1\n                (block-params-c-attn-w params)\n                (block-params-c-attn-b params)))\n      (lambda (y t0) (trace-activation! obs 'block-qkv y t0))))"
  },
  "kv-write": {
    "file": "src/block.rkt",
    "fn": "write-qkv!",
    "startLine": 117,
    "highlightStart": 122,
    "highlightEnd": 125,
    "callSiteLine": 173,
    "callSiteEndLine": 173,
    "code": "(define (write-qkv! k v row mixed width)\n  (define from (tensor-data mixed))\n  (define kd (tensor-data k))\n  (define vd (tensor-data v))\n  (for ([i (in-range (car (tensor-dims mixed)))])\n    (define src (+ (* i 3 width) width))\n    (define dest (* (+ row i) width))\n    (for ([j (in-range width)])\n      (flvector-set! kd (+ dest j) (flvector-ref from (+ src j)))\n      (flvector-set! vd (+ dest j) (flvector-ref from (+ src width j))))))"
  },
  "attend": {
    "file": "src/attention.rkt",
    "fn": "fill-head-fast!",
    "startLine": 29,
    "highlightStart": 36,
    "highlightEnd": 73,
    "callSiteLine": 232,
    "callSiteEndLine": 234,
    "code": "(define (fill-head-fast! out q k v scores h nq nk q-stride stride dh capacity scale)\n  (define col (unsafe-fx* h dh))\n  (define score-base (unsafe-fx* h capacity))\n  (let loop-i ([i 0])\n    (unless (unsafe-fx= i nq)\n      (define qbase (unsafe-fx+ (unsafe-fx* i q-stride) col))\n      (define outbase (unsafe-fx+ (unsafe-fx* i stride) col))\n      (define visible (unsafe-fx+ (unsafe-fx- nk nq) (unsafe-fx+ i 1)))\n      (define maxv\n        (let loop-j ([j 0] [maxv -inf.0])\n          (if (unsafe-fx= j visible)\n              maxv\n              (let* ([kbase (unsafe-fx+ (unsafe-fx* j stride) col)]\n                     [dot (let loop-d ([d 0] [sum 0.0])\n                            (if (unsafe-fx= d dh) sum\n                                (loop-d (unsafe-fx+ d 1)\n                                        (unsafe-fl+ sum\n                                                    (unsafe-fl* (unsafe-flvector-ref q (unsafe-fx+ qbase d))\n                                                                (unsafe-flvector-ref k (unsafe-fx+ kbase d)))))))]\n                     [score (unsafe-fl* dot scale)])\n                (unsafe-flvector-set! scores (unsafe-fx+ score-base j) score)\n                (loop-j (unsafe-fx+ j 1) (flmax maxv score))))))\n      (define total\n        (let loop-j ([j 0] [sum 0.0])\n          (if (unsafe-fx= j visible) sum\n              (let* ([idx (unsafe-fx+ score-base j)]\n                     [score (unsafe-flvector-ref scores idx)]\n                     [weight (cond [(fl= maxv -inf.0) 0.0]\n                                   [(fl= maxv +inf.0) (if (fl= score +inf.0) 1.0 0.0)]\n                                   [else (flexp (unsafe-fl- score maxv))])])\n                (unsafe-flvector-set! scores idx weight)\n                (loop-j (unsafe-fx+ j 1) (unsafe-fl+ sum weight))))))\n      (unless (fl= total 0.0)\n        (let loop-j ([j 0])\n          (unless (unsafe-fx= j visible)\n            (define weight (unsafe-fl/ (unsafe-flvector-ref scores (unsafe-fx+ score-base j)) total))\n            (define vbase (unsafe-fx+ (unsafe-fx* j stride) col))\n            (let loop-d ([d 0])\n              (unless (unsafe-fx= d dh)\n                (define idx (unsafe-fx+ outbase d))\n                (unsafe-flvector-set! out idx\n                                      (unsafe-fl+ (unsafe-flvector-ref out idx)\n                                                  (unsafe-fl* weight (unsafe-flvector-ref v (unsafe-fx+ vbase d)))))\n                (loop-d (unsafe-fx+ d 1))))\n            (loop-j (unsafe-fx+ j 1)))))\n      (loop-i (unsafe-fx+ i 1)))))"
  },
  "block-proj-res": {
    "file": "src/block.rkt",
    "fn": "matmul-add",
    "startLine": 193,
    "highlightStart": 196,
    "highlightEnd": 199,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "  (define residual1\n    (if-trace obs 'block-proj-res\n      (stage block-proj-res matmul-add\n        (matmul-add heads\n                    (block-params-attn-proj-w params)\n                    (block-params-attn-proj-b params)\n                    x))\n      (lambda (y t0) (trace-activation! obs 'block-proj-res y t0))))"
  },
  "block-ln2": {
    "file": "src/block.rkt",
    "fn": "layer-norm",
    "startLine": 201,
    "highlightStart": 204,
    "highlightEnd": 206,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "  (define ln2\n    (if-trace obs 'block-ln2\n      (stage block-ln2 layer-norm\n        (layer-norm residual1\n                    (block-params-ln2-gain params)\n                    (block-params-ln2-bias params)))\n      (lambda (y t0) (trace-activation! obs 'block-ln2 y t0))))"
  },
  "mlp-gelu": {
    "file": "src/block.rkt",
    "fn": "gpt2-mlp-hidden",
    "startLine": 208,
    "highlightStart": 211,
    "highlightEnd": 213,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "  (define hidden\n    (if-trace obs 'mlp-gelu\n      (stage mlp-gelu gpt2-mlp-hidden\n        (gpt2-mlp-hidden ln2\n                         (block-params-c-fc-w params)\n                         (block-params-c-fc-b params)))\n      (lambda (y t0) (trace-activation! obs 'mlp-gelu y t0))))"
  },
  "block-mlp-res": {
    "file": "src/block.rkt",
    "fn": "matmul-add",
    "startLine": 215,
    "highlightStart": 218,
    "highlightEnd": 221,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "  (define out\n    (if (eq? obs #f)\n        (stage block-mlp-res matmul-add\n          (matmul-add hidden\n                      (block-params-mlp-proj-w params)\n                      (block-params-mlp-proj-b params)\n                      residual1))"
  },
  "final-ln": {
    "file": "src/model.rkt",
    "fn": "layer-norm",
    "startLine": 184,
    "highlightStart": 187,
    "highlightEnd": 187,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "    (define hidden\n      (if-trace obs 'final-ln\n        (stage final-ln layer-norm\n          (layer-norm y (gpt2-model-ln-gain model) (gpt2-model-ln-bias model)))\n        (lambda (h t0) (trace-activation! obs 'final-ln h t0))))"
  },
  "tied-logits": {
    "file": "src/model.rkt",
    "fn": "tied-logits",
    "startLine": 193,
    "highlightStart": 196,
    "highlightEnd": 196,
    "callSiteLine": null,
    "callSiteEndLine": null,
    "code": "    (define logits\n      (if-trace obs 'tied-logits\n        (stage tied-logits tied-logits\n          (tied-logits scored (gpt2-model-wte model)))\n        (lambda (z t0) (trace-activation! obs 'tied-logits z t0))))"
  },
  "sample": {
    "file": "src/model.rkt",
    "fn": "argmax",
    "startLine": 241,
    "highlightStart": 241,
    "highlightEnd": 249,
    "callSiteLine": 324,
    "callSiteEndLine": 324,
    "code": "(define (argmax logits)\n  (define n (flvector-length logits))\n  (let loop ([i 1] [best 0] [bestv (flvector-ref logits 0)])\n    (if (= i n)\n        best\n        (let ([v (flvector-ref logits i)])\n          (if (fl> v bestv)\n              (loop (add1 i) i v)\n              (loop (add1 i) best bestv))))))"
  }
};
