#lang racket/base

;; One pre-LN GPT-2 block: h = x + attn(ln_1(x)); out = h + mlp(ln_2(h)).
;; The MLP is c_fc [width, inner], tanh GELU, c_proj [inner, width].

(require "attention.rkt"
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

;; past-k and past-v are earlier rows, or #f. The new rows are appended
;; and the returned k and v are the whole cache for this layer.
(define (gpt2-block-cached x params past-k past-v)
  (unless (block-params? params)
    (raise-argument-error 'gpt2-block-cached "block-params?" params))
  (when (or (and past-k (not past-v)) (and past-v (not past-k)))
    (raise-arguments-error 'gpt2-block-cached
                           "k and v caches must both be present"
                           "k" past-k
                           "v" past-v))
  (define ln1 (layer-norm x
                          (block-params-ln1-gain params)
                          (block-params-ln1-bias params)))
  (define mixed (matmul ln1
                        (block-params-c-attn-w params)
                        (block-params-c-attn-b params)))
  (define-values (q k v) (qkv-split mixed))
  (define kall (if past-k (tensor-append-rows past-k k) k))
  (define vall (if past-v (tensor-append-rows past-v v) v))
  (define heads (attend q kall vall (block-params-n-head params)))
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
  (values (tensor-add residual1 mlp) kall vall))

(define (gpt2-blocks x params-list)
  (unless (list? params-list)
    (raise-argument-error 'gpt2-blocks "list?" params-list))
  (for/fold ([hidden x]) ([params (in-list params-list)])
    (gpt2-block hidden params)))
