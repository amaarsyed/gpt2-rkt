#lang racket/base

(module+ test
  (require rackunit
           "../src/bpe-bytes.rkt")

  (test-case "byte map round-trips every byte"
    (define seen (make-hash))
    (for ([b (in-range bpe-byte-count)])
      (define c (bpe-byte->char b))
      (check-equal? (bpe-char->byte c) b)
      (check-false (hash-has-key? seen c))
      (hash-set! seen c #t))
    (check-equal? (hash-count seen) bpe-byte-count))

  (test-case "printable bytes stay themselves and space maps to U+0120"
    (check-equal? (bpe-byte->char 33) #\!)
    (check-equal? (bpe-byte->char 126) #\~)
    (check-equal? (char->integer (bpe-byte->char 32)) #x120)
    (check-false (char=? (bpe-byte->char 0) (integer->char 0)))
    (check-false (char=? (bpe-byte->char 10) #\newline))
    (check-exn exn:fail:contract?
               (lambda () (bpe-char->byte #\u4E00))))
  )