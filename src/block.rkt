#lang racket/base

;; One pre-LN GPT-2 block: h = x + attn(ln_1(x)); out = h + mlp(ln_2(h)).
;; The MLP is c_fc [width, inner], tanh GELU, c_proj [inner, width].

(require racket/flonum
         "attention.rkt"
         "nn.rkt"
         "tensor.rkt")

(provide gpt2-mlp
         gpt2-mlp-hidden
         make-block-params
         block-params-from-tensors
         block-params?
         block-params-n-head
         block-trace?
         block-trace-ln1
         block-trace-attn
         block-trace-residual1
         block-trace-ln2
         block-trace-mlp-hidden
         block-trace-out
         gpt2-block-parts
         gpt2-block
         gpt2-block-cached
         gpt2-blocks)

(struct block-params (n-head
                      ln1-gain ln1-bias
                      c-attn-w c-attn-b
                      attn-proj-w attn-proj-b
                      ln2-gain ln2-bias
                      c-fc-w c-fc-b
                      mlp-proj-w mlp-proj-b)
  #:constructor-name raw-block-params
  #:transparent)

;; Keywords, not positions: a gain/bias swap has the same shape.
;; block-params-from-tensors uses the Hugging Face suffixes so a loader
;; can pass one layer's tensors without reordering fields.
(define (make-block-params #:n-head n-head
                           #:ln1-gain ln1-gain
                           #:ln1-bias ln1-bias
                           #:c-attn-w c-attn-w
                           #:c-attn-b c-attn-b
                           #:attn-proj-w attn-proj-w
                           #:attn-proj-b attn-proj-b
                           #:ln2-gain ln2-gain
                           #:ln2-bias ln2-bias
                           #:c-fc-w c-fc-w
                           #:c-fc-b c-fc-b
                           #:mlp-proj-w mlp-proj-w
                           #:mlp-proj-b mlp-proj-b)
  (raw-block-params n-head
                    ln1-gain ln1-bias
                    c-attn-w c-attn-b
                    attn-proj-w attn-proj-b
                    ln2-gain ln2-bias
                    c-fc-w c-fc-b
                    mlp-proj-w mlp-proj-b))

(define (block-params-from-tensors #:n-head n-head tensors)
  (unless (hash? tensors)
    (raise-argument-error 'block-params-from-tensors "hash?" tensors))
  (define (get name)
    (hash-ref tensors name
              (lambda ()
                (raise-arguments-error 'block-params-from-tensors
                                       "missing tensor"
                                       "name" name))))
  (make-block-params #:n-head n-head
                     #:ln1-gain (get "ln_1.weight")
                     #:ln1-bias (get "ln_1.bias")
                     #:c-attn-w (get "attn.c_attn.weight")
                     #:c-attn-b (get "attn.c_attn.bias")
                     #:attn-proj-w (get "attn.c_proj.weight")
                     #:attn-proj-b (get "attn.c_proj.bias")
                     #:ln2-gain (get "ln_2.weight")
                     #:ln2-bias (get "ln_2.bias")
                     #:c-fc-w (get "mlp.c_fc.weight")
                     #:c-fc-b (get "mlp.c_fc.bias")
                     #:mlp-proj-w (get "mlp.c_proj.weight")
                     #:mlp-proj-b (get "mlp.c_proj.bias")))

(struct block-trace (ln1 attn residual1 ln2 mlp-hidden out)
  #:transparent)

(define (gpt2-mlp-hidden x c-fc-w c-fc-b)
  (gelu (matmul x c-fc-w c-fc-b)))

(define (gpt2-mlp x c-fc-w c-fc-b c-proj-w c-proj-b)
  (matmul (gpt2-mlp-hidden x c-fc-w c-fc-b) c-proj-w c-proj-b))

(define (gpt2-block-parts x params)
  (unless (block-params? params)
    (raise-argument-error 'gpt2-block "block-params?" params))
  (define ln1 (layer-norm x
                          (block-params-ln1-gain params)
                          (block-params-ln1-bias params)))
  (define attn (gpt2-attention ln1
                               (block-params-c-attn-w params)
                               (block-params-c-attn-b params)
                               (block-params-attn-proj-w params)
                               (block-params-attn-proj-b params)
                               (block-params-n-head params)))
  (define residual1 (tensor-add x attn))
  (define ln2 (layer-norm residual1
                          (block-params-ln2-gain params)
                          (block-params-ln2-bias params)))
  (define hidden (gpt2-mlp-hidden ln2
                                  (block-params-c-fc-w params)
                                  (block-params-c-fc-b params)))
  (define mlp (matmul hidden
                      (block-params-mlp-proj-w params)
                      (block-params-mlp-proj-b params)))
  (block-trace ln1 attn residual1 ln2 hidden (tensor-add residual1 mlp)))

(define (gpt2-block x params)
  (block-trace-out (gpt2-block-parts x params)))

(define (write-rows! dest row src)
  (define cols (cadr (tensor-dims dest)))
  (define n (car (tensor-dims src)))
  (define from (tensor-data src))
  (define to (tensor-data dest))
  (define base (* row cols))
  (for ([i (in-range (* n cols))])
    (flvector-set! to (+ base i) (flvector-ref from i))))

;; past-k and past-v are preallocated buffers. The new rows are written
;; at #:filled, and attend sees only the filled prefix.
(define (gpt2-block-cached x params past-k past-v #:filled filled #:workspace [workspace #f])
  (unless (block-params? params)
    (raise-argument-error 'gpt2-block-cached "block-params?" params))
  (unless (and (tensor? past-k) (tensor? past-v))
    (raise-arguments-error 'gpt2-block-cached
                           "k and v caches must be tensors"
                           "k" past-k
                           "v" past-v))
  (unless (exact-nonnegative-integer? filled)
    (raise-argument-error 'gpt2-block-cached "exact-nonnegative-integer?" filled))
  (define ln1 (layer-norm x
                          (block-params-ln1-gain params)
                          (block-params-ln1-bias params)))
  (define mixed (matmul ln1
                        (block-params-c-attn-w params)
                        (block-params-c-attn-b params)))
  (define-values (q k v) (qkv-split mixed))
  (define nq (car (tensor-dims k)))
  (define cap (car (tensor-dims past-k)))
  (unless (and (= cap (car (tensor-dims past-v)))
               (<= (+ filled nq) cap))
    (raise-arguments-error 'gpt2-block-cached
                           "the cache is shorter than the next rows"
                           "filled" filled
                           "new" nq
                           "capacity" cap))
  (write-rows! past-k filled k)
  (write-rows! past-v filled v)
  (define kv-rows (+ filled nq))
  (define heads (attend q past-k past-v (block-params-n-head params)
                        #:kv-rows kv-rows #:workspace workspace))
  (define attn (matmul heads
                       (block-params-attn-proj-w params)
                       (block-params-attn-proj-b params)))
  (define residual1 (tensor-add x attn))
  (define ln2 (layer-norm residual1
                          (block-params-ln2-gain params)
                          (block-params-ln2-bias params)))
  (define mlp (gpt2-mlp ln2
                        (block-params-c-fc-w params)
                        (block-params-c-fc-b params)
                        (block-params-mlp-proj-w params)
                        (block-params-mlp-proj-b params)))
  (values (tensor-add residual1 mlp) past-k past-v))

(define (gpt2-blocks x params-list)
  (unless (list? params-list)
    (raise-argument-error 'gpt2-blocks "list?" params-list))
  (for/fold ([hidden x]) ([params (in-list params-list)])
    (gpt2-block hidden params)))
