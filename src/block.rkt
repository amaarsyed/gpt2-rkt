#lang racket/base

;; One pre-LN GPT-2 block: x + attn(ln_1(x)), then that plus mlp(ln_2(x)).
;; The MLP is c_fc [width, inner], tanh GELU, c_proj [inner, width].

(require "attention.rkt"
         "nn.rkt"
         "tensor.rkt")

(provide gpt2-mlp
         gpt2-mlp-hidden
         make-block-params
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
         gpt2-block)

(struct block-params (n-head
                      ln1-gain ln1-bias
                      c-attn-w c-attn-b
                      attn-proj-w attn-proj-b
                      ln2-gain ln2-bias
                      c-fc-w c-fc-b
                      mlp-proj-w mlp-proj-b)
  #:constructor-name make-block-params
  #:transparent)

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
