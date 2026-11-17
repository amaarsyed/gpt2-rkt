#lang racket/base

;; GPT-2 MLP. c_fc is [width, inner] and c_proj is [inner, width].
;; The activation is the tanh GELU, not the erf GELU.

(require "nn.rkt"
         "tensor.rkt")

(provide gpt2-mlp
         gpt2-mlp-hidden)

(define (gpt2-mlp-hidden x c-fc-w c-fc-b)
  (gelu (matmul x c-fc-w c-fc-b)))

(define (gpt2-mlp x c-fc-w c-fc-b c-proj-w c-proj-b)
  (matmul (gpt2-mlp-hidden x c-fc-w c-fc-b) c-proj-w c-proj-b))
