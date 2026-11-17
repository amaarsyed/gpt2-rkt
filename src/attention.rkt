#lang racket/base

;; GPT-2 causal attention in the Hugging Face Conv1D layout.
;;
;; c_attn yields [n, 3*d]. That axis is split into q, k, and v, each
;; [n, d]. Head h is the contiguous column block [h*d-head, (h+1)*d-head).
;; Scores are scaled by 1/sqrt(d-head). A key strictly after the query's
;; position is -inf, then softmax, then the merge and c_proj.
;;
;; KV cache: k and v may have more rows than q. Query row i sits at
;; absolute position (rows(k) - rows(q) + i). #:kv-rows limits that
;; count when the buffer is longer than the filled prefix. Each head
;; reads its columns from that prefix in place, with a column offset,
;; instead of copying the prefix out.

(require racket/flonum
         "nn.rkt"
         "tensor.rkt")

(provide qkv-split
         attention-scale
         apply-causal-mask!
         causal-scores
         attend
         gpt2-attention)

(define (attention-scale d-head)
  (unless (exact-positive-integer? d-head)
    (raise-argument-error 'attention-scale "exact-positive-integer?" d-head))
  (fl/ 1.0 (flsqrt (real->double-flonum d-head))))

(define (qkv-split mixed)
  (define parts (split-last mixed 3))
  (values (car parts) (cadr parts) (caddr parts)))

(define (apply-causal-mask! scores scale)
  (define nq (car (tensor-dims scores)))
  (define nk (cadr (tensor-dims scores)))
  (define src (tensor-data scores))
  (define pos0 (- nk nq))
  (for ([i (in-range nq)])
    (define limit (+ pos0 i))
    (for ([j (in-range nk)])
      (define idx (+ (* i nk) j))
      (flvector-set! src idx
                     (if (> j limit)
                         -inf.0
                         (fl* scale (flvector-ref src idx))))))
  scores)

(define (causal-scores q k scale)
  (unless (and (flonum? scale) (fl> scale 0.0))
    (raise-argument-error 'causal-scores "positive flonum?" scale))
  (define-values (nq dh-q) (matrix-shape 'causal-scores q))
  (define-values (nk dh-k) (matrix-shape 'causal-scores k))
  (unless (= dh-q dh-k)
    (raise-arguments-error 'causal-scores
                           "q and k head widths differ"
                           "q" (dims->string (tensor-dims q))
                           "k" (dims->string (tensor-dims k))))
  (when (> nq nk)
    (raise-arguments-error 'causal-scores
                           "q is longer than k"
                           "q rows" nq
                           "k rows" nk))
  (apply-causal-mask! (matmul-transposed-b q k) scale))

(define (attend q k v n-head #:kv-rows [kv-rows #f])
  (unless (exact-positive-integer? n-head)
    (raise-argument-error 'attend "exact-positive-integer?" n-head))
  (define-values (nq d) (matrix-shape 'attend q))
  (define-values (nk dk) (matrix-shape 'attend k))
  (define-values (nv dv) (matrix-shape 'attend v))
  (unless (and (= d dk) (= d dv))
    (raise-arguments-error 'attend
                           "q, k, and v widths differ"
                           "q" (dims->string (tensor-dims q))
                           "k" (dims->string (tensor-dims k))
                           "v" (dims->string (tensor-dims v))))
  (unless (= nk nv)
    (raise-arguments-error 'attend
                           "k and v row counts differ"
                           "k rows" nk
                           "v rows" nv))
  (define used (or kv-rows nk))
  (unless (and (exact-positive-integer? used) (<= used nk))
    (raise-arguments-error 'attend
                           "kv row count is outside the cache"
                           "rows" used
                           "k rows" nk))
  (when (> nq used)
    (raise-arguments-error 'attend
                           "q is longer than k"
                           "q rows" nq
                           "k rows" used))
  (unless (zero? (remainder d n-head))
    (raise-arguments-error 'attend
                           "width is not divisible by the head count"
                           "width" d
                           "heads" n-head))
  (define dh (quotient d n-head))
  (define scale (attention-scale dh))
  (define bias (make-tensor (list dh) (make-flvector dh 0.0)))
  (define merged (make-tensor (list nq d) (make-flvector (* nq d) 0.0)))
  (for ([h (in-range n-head)])
    (define start (* h dh))
    (define qh (slice-cols q start (+ start dh)))
    (define scores (matmul-transposed-block qh k used start))
    (define probs (softmax (apply-causal-mask! scores scale)))
    (paste-cols! merged d (matmul-column-block probs v bias start dh) start))
  merged)

(define (gpt2-attention x c-attn-w c-attn-b c-proj-w c-proj-b n-head)
  (unless (exact-positive-integer? n-head)
    (raise-argument-error 'gpt2-attention "exact-positive-integer?" n-head))
  (define mixed (matmul x c-attn-w c-attn-b))
  (define-values (q k v) (qkv-split mixed))
  (define heads (attend q k v n-head))
  (matmul heads c-proj-w c-proj-b))
