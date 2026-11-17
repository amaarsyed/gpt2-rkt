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
         racket/future
         racket/unsafe/ops
         "nn.rkt"
         "tensor.rkt")

(provide qkv-split
         make-attention-work
         attention-scale
         apply-causal-mask!
         causal-scores
         attend
         gpt2-attention)

;; Scores belong to one request. Heads use disjoint portions, and each
;; query reuses its head's row after the previous output is complete.
(struct attention-work (heads capacity scores) #:transparent)

(define (make-attention-work heads capacity)
  (unless (and (exact-positive-integer? heads) (exact-positive-integer? capacity))
    (raise-arguments-error 'make-attention-work "heads and capacity must be positive"
                           "heads" heads "capacity" capacity))
  (attention-work heads capacity (make-flvector (* heads capacity) 0.0)))

(define (fill-head! out q k v scores h nq nk stride dh capacity scale)
  (define col (unsafe-fx* h dh))
  (define score-base (unsafe-fx* h capacity))
  (let loop-i ([i 0])
    (unless (unsafe-fx= i nq)
      (define qbase (unsafe-fx+ (unsafe-fx* i stride) col))
      (define visible (unsafe-fx+ (unsafe-fx- nk nq) (unsafe-fx+ i 1)))
      (define maxv
        (let loop-j ([j 0] [maxv -inf.0])
          (if (unsafe-fx= j visible)
              maxv
              (let* ([kbase (unsafe-fx+ (unsafe-fx* j stride) col)]
                     [dot (let loop-d ([d 0] [sum 0.0])
                            (if (unsafe-fx= d dh) sum
                                (loop-d (unsafe-fx+ d 1)
                                        (unsafe-fl+ sum
                                                    (unsafe-fl* (unsafe-flvector-ref q (unsafe-fx+ qbase d))
                                                                (unsafe-flvector-ref k (unsafe-fx+ kbase d)))))))]
                     [score (unsafe-fl* dot scale)])
                (unsafe-flvector-set! scores (unsafe-fx+ score-base j) score)
                (loop-j (unsafe-fx+ j 1) (flmax maxv score))))))
      (define total
        (let loop-j ([j 0] [sum 0.0])
          (if (unsafe-fx= j visible) sum
              (let* ([idx (unsafe-fx+ score-base j)]
                     [score (unsafe-flvector-ref scores idx)]
                     [weight (cond [(fl= maxv -inf.0) 0.0]
                                   [(fl= maxv +inf.0) (if (fl= score +inf.0) 1.0 0.0)]
                                   [else (flexp (unsafe-fl- score maxv))])])
                (unsafe-flvector-set! scores idx weight)
                (loop-j (unsafe-fx+ j 1) (unsafe-fl+ sum weight))))))
      (unless (fl= total 0.0)
        (let loop-j ([j 0])
          (unless (unsafe-fx= j visible)
            (define weight (unsafe-fl/ (unsafe-flvector-ref scores (unsafe-fx+ score-base j)) total))
            (define vbase (unsafe-fx+ (unsafe-fx* j stride) col))
            (let loop-d ([d 0])
              (unless (unsafe-fx= d dh)
                (define idx (unsafe-fx+ qbase d))
                (unsafe-flvector-set! out idx
                                      (unsafe-fl+ (unsafe-flvector-ref out idx)
                                                  (unsafe-fl* weight (unsafe-flvector-ref v (unsafe-fx+ vbase d)))))
                (loop-d (unsafe-fx+ d 1))))
            (loop-j (unsafe-fx+ j 1)))))
      (loop-i (unsafe-fx+ i 1)))))

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

(define (attend q k v n-head #:kv-rows [kv-rows #f] #:workspace [workspace #f])
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
  (define work (or workspace (make-attention-work n-head used)))
  (unless (and (attention-work? work)
               (>= (attention-work-heads work) n-head)
               (>= (attention-work-capacity work) used))
    (raise-arguments-error 'attend "workspace does not fit the heads and filled prefix"
                           "heads" n-head "rows" used))
  (define merged (make-tensor (list nq d) (make-flvector (* nq d) 0.0)))
  (define (head h)
    (fill-head! (tensor-data merged) (tensor-data q) (tensor-data k) (tensor-data v)
                (attention-work-scores work) h nq used d dh (attention-work-capacity work) scale))
  ;; A decode query stays serial. Prefill gives each head its own future.
  (cond
    [(and (> (processor-count) 1) (> (* nq used dh) 100000))
     (define jobs (for/list ([h (in-range n-head)]) (future (lambda () (head h)))))
     (for ([job (in-list jobs)]) (touch job))]
    [else (for ([h (in-range n-head)]) (head h))])
  merged)

(define (gpt2-attention x c-attn-w c-attn-b c-proj-w c-proj-b n-head)
  (unless (exact-positive-integer? n-head)
    (raise-argument-error 'gpt2-attention "exact-positive-integer?" n-head))
  (define mixed (matmul x c-attn-w c-attn-b))
  (define-values (q k v) (qkv-split mixed))
  (define heads (attend q k v n-head))
  (matmul heads c-proj-w c-proj-b))
