#lang racket/base

;; User-facing failures for the CLI, loader, and tokenizer.

(provide exn:fail:gpt2?
         raise-gpt2)

(struct exn:fail:gpt2 exn:fail ()
  #:transparent)

(define (raise-gpt2 message)
  (raise (exn:fail:gpt2 message (current-continuation-marks))))
