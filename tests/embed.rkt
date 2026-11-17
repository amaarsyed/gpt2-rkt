#lang racket/base

(module+ test
  (require rackunit
           racket/flonum
           "../src/embed.rkt"
           "../src/tensor.rkt")

  (define (elements t)
    (define data (tensor-data t))
    (for/list ([i (in-range (flvector-length data))])
      (flvector-ref data i)))

  (define wte
    (make-tensor '(4 2) (flvector 1.0 2.0
                                  3.0 4.0
                                  5.0 6.0
                                  7.0 8.0)))
  (define wpe
    (make-tensor '(4 2) (flvector 0.5 0.0
                                  0.0 0.5
                                  1.0 1.0
                                  9.0 9.0)))

  (test-case "token embed copies the selected rows"
    (define y (token-embed wte '(2 0 2)))
    (check-equal? (tensor-dims y) '(3 2))
    (check-equal? (elements y) '(5.0 6.0 1.0 2.0 5.0 6.0))
    (check-equal? (elements (token-embed wte '())) '())
    (check-exn #rx"out of range"
               (lambda () (token-embed wte '(4))))
    (check-exn exn:fail:contract?
               (lambda () (token-embed wte '(-1)))))

  (test-case "position embed is 0 .. n-1 and stops at 1024"
    (check-equal? gpt2-context-length 1024)
    (define y (position-embed wpe 3))
    (check-equal? (elements y) '(0.5 0.0 0.0 0.5 1.0 1.0))
    (check-exn #rx"position table"
               (lambda () (position-embed wpe 5)))
    (define big (make-flvector 1025 0.0))
    (check-exn #rx"1024"
               (lambda ()
                 (position-embed (make-tensor '(1025 1) big) 1025)))
    (define rows (make-flvector 1024))
    (for ([i (in-range 1024)])
      (flvector-set! rows i (exact->inexact i)))
    (define table (make-tensor '(1024 1) rows))
    (define full (position-embed table 1024))
    (check-equal? (tensor-dims full) '(1024 1))
    (check-equal? (tensor-ref full 0 0) 0.0)
    (check-equal? (tensor-ref full 1023 0) 1023.0)
    (check-equal? (elements (position-embed table 1 #:start 1023)) '(1023.0))
    (check-exn #rx"1024"
               (lambda () (position-embed table 1 #:start 1024)))
    (check-exn #rx"1024"
               (lambda () (position-embed table 2 #:start 1023))))

  (test-case "position embed reads rows from a start offset"
    (define y (position-embed wpe 2 #:start 1))
    (check-equal? (tensor-dims y) '(2 2))
    (check-equal? (elements y) '(0.0 0.5 1.0 1.0))
    (check-equal? (elements (input-embed wte wpe '(2) #:start 1))
                  '(5.0 6.5))
    (check-exn exn:fail:contract?
               (lambda () (position-embed wpe 1 #:start -1))))

  (test-case "input embed adds token and position rows"
    (define y (input-embed wte wpe '(1 2)))
    (check-equal? (elements y) '(3.5 4.0 5.0 6.5))
    (check-equal? (tensor-dims (input-embed wte wpe '())) '(0 2)))
  )