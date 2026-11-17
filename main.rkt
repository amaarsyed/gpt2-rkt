#lang racket/base

;; CLI entry and public API. Implementation lives under src/.

(require "src/cli.rkt"
         "src/errors.rkt"
         "src/generate.rkt"
         "src/model-assets.rkt"
         "src/nn.rkt"
         "src/tensor.rkt")

(provide run-cli
         parse-generate-args
         top-help-text
         generate-help-text
         generate
         generate-config
         generate-config?
         generate-config-model-dir
         generate-config-prompt
         generate-config-max-new-tokens
         generate-config-temperature
         generate-config-top-k
         generate-config-seed
         exn:fail:gpt2?
         required-model-asset-names
         make-tensor
         tensor?
         tensor-dims
         tensor-data
         tensor-ref
         tensor-add
         slice-rows
         slice-cols
         split-last
         tensor-reshape
         flvector-dot
         matmul
         matmul-transposed-b
         softmax
         layer-norm
         layernorm-eps
         gelu)

(module+ main
  (exit (run-cli (current-command-line-arguments))))
