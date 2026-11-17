#lang info

(define collection "gpt2-rkt")
(define version "0.1.0")
(define deps '("base"))
(define build-deps '("rackunit-lib"))
(define pkg-desc "Pure-Racket GPT-2 124M inference")
(define pkg-authors '(amaarsyed))
;; Non-Racket directories are omitted from compilation and tests.
(define test-omit-paths '("info.rkt" "tools" "docs" ".github" "models"))
(define compile-omit-paths '("tools" "docs" ".github" "models"))
