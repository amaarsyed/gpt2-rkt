#lang racket/base

;; CLI entry and public API. Implementation lives under src/.

(require "src/attention.rkt"
         "src/block.rkt"
         "src/bpe-bytes.rkt"
         "src/cli.rkt"
         "src/embed.rkt"
         "src/errors.rkt"
         "src/generate.rkt"
         "src/model.rkt"
         "src/model-assets.rkt"
         "src/nn.rkt"
         "src/pretokenize.rkt"
         "src/tensor.rkt"
         "src/tokenizer.rkt")

(provide run-cli
         parse-generate-args
         top-help-text
         generate-help-text
         generate
         load-model
         next-token-logits
         generation?
         generation-token-ids
         generation-text
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
         gelu
         bpe-byte->char
         bpe-char->byte
         gpt2-pretokenize
         load-tokenizer
         make-tokenizer
         tokenizer?
         encode
         decode
         endoftext-token
         endoftext-id
         gpt2-context-length
         token-embed
         position-embed
         input-embed
         qkv-split
         attention-scale
         causal-scores
         attend
         gpt2-attention
         gpt2-mlp
         gpt2-block
         gpt2-blocks)

(module+ main
  (exit (run-cli (current-command-line-arguments))))
