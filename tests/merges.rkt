#lang racket/base

(module+ test
  (require rackunit
           "../src/errors.rkt"
           "../src/merges.rkt")

  (test-case "the #version line is not a merge"
    (define ranks (read-merge-ranks "#version: 0.2\na b\nĠ t\n"))
    (check-equal? (hash-ref ranks (cons "a" "b")) 0)
    (check-equal? (hash-ref ranks (cons "Ġ" "t")) 1)
    (check-false (hash-has-key? ranks (cons "#version:" "0.2")))
    (check-equal? (hash-count ranks) 2))

  (test-case "a malformed merge line is an error"
    (check-exn exn:fail:gpt2?
               (lambda () (read-merge-ranks "a b c\n"))))
  )