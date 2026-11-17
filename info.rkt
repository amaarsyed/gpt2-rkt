#lang info

(define collection "gpt2-rkt")
(define version "0.1.0")
(define deps '("base"))
(define build-deps '("rackunit-lib"))
(define pkg-desc "Pure-Racket GPT-2 124M inference")
(define pkg-authors '(amaarsyed))
;; Python fixtures and docs are not Racket modules.
(define test-omit-paths '("info.rkt" "tools" "docs" ".github"))
(define compile-omit-paths '("tools" "docs" ".github"))
