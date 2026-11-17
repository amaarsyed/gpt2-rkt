#lang racket/base

;; CLI entry and public API. Implementation lives under src/.

(require "src/cli.rkt"
         "src/generate.rkt"
         "src/model-assets.rkt")

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
         required-model-asset-names)

(module+ main
  (exit (run-cli (current-command-line-arguments))))
