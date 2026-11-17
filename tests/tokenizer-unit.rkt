#lang racket/base

(module+ test
  (require rackunit
           "../src/errors.rkt"
           "../src/merges.rkt"
           "../src/tokenizer.rkt")

  (define (tiny)
    (make-tokenizer (hash "a" 0 "b" 1 "c" 2 "bc" 3 "abc" 4 "ab" 5)
                    (read-merge-ranks "#version: 0.2\nb c\na bc\na b\n")))

  (test-case "BPE applies the lowest merge rank, ignoring #version"
    (define tok (tiny))
    (check-equal? (encode tok "abc") '(4))
    (check-equal? (encode tok "ab") '(5))
    (check-equal? (encode tok "ba") '(1 0))
    (check-equal? (encode tok "") '())
    (check-equal? (decode tok (encode tok "abc")) "abc")
    (check-equal? (encode tok "abc") '(4)))

  (test-case "decode rejects an unknown id"
    (check-exn #rx"unknown token id: 9"
               (lambda () (decode (tiny) '(9)))))

  (test-case "<|endoftext|> is one token when the vocab has it"
    (define tok
      (make-tokenizer (hash "a" 0
                            "b" 1
                            "<|endoftext|>" 50256)
                      (hash)))
    (check-equal? (encode tok "a<|endoftext|>b") '(0 50256 1))
    (check-equal? (decode tok '(50256)) "<|endoftext|>")
    (check-equal? endoftext-id 50256))
  )