#lang racket/base

(module+ test
  (require rackunit
           racket/flonum
           "../src/tensor.rkt")

  (define (elements t)
    (define data (tensor-data t))
    (for/list ([i (in-range (flvector-length data))])
      (flvector-ref data i)))

  (test-case "tensor stores dims and row-major elements"
    (define t (make-tensor '(2 3) (flvector 1.0 2.0 3.0 4.0 5.0 6.0)))
    (check-equal? (tensor-dims t) '(2 3))
    (check-equal? (tensor-ref t 1 2) 6.0)
    (check-equal? (tensor-ref t 0 0) 1.0))

  (test-case "make-tensor rejects a length that does not match dims"
    (check-exn #rx"\\(2 3\\)"
               (lambda ()
                 (make-tensor '(2 3) (flvector 1.0))))
    (check-exn exn:fail:contract?
               (lambda ()
                 (make-tensor '(2) '#(1.0 2.0))))
    (check-exn exn:fail:contract?
               (lambda ()
                 (make-tensor '() (flvector)))))

  (test-case "tensor-ref rejects a bad index"
    (define t (make-tensor '(2 2) (flvector 1.0 2.0 3.0 4.0)))
    (check-exn #rx"out of range"
               (lambda () (tensor-ref t 2 0)))
    (check-exn #rx"rank"
               (lambda () (tensor-ref t 0))))

  (test-case "tensor-add is the residual add"
    (define a (make-tensor '(2 2) (flvector 1.0 2.0 3.0 4.0)))
    (define b (make-tensor '(2 2) (flvector 10.0 -2.0 0.5 1.0)))
    (check-equal? (elements (tensor-add a b)) '(11.0 0.0 3.5 5.0))
    (check-exn #rx"\\(2 2\\)"
               (lambda ()
                 (tensor-add a (make-tensor '(4) (flvector 1.0 2.0 3.0 4.0))))))

  (test-case "reshape shares the flvector and slice copies"
    (define t (make-tensor '(2 3) (flvector 1.0 2.0 3.0 4.0 5.0 6.0)))
    (define reshaped (tensor-reshape t '(3 2)))
    (check-eq? (tensor-data reshaped) (tensor-data t))
    (check-equal? (tensor-dims reshaped) '(3 2))
    (check-equal? (tensor-ref reshaped 1 0) 3.0)
    (define rows (slice-rows t 1 2))
    (define cols (slice-cols t 1 3))
    (check-equal? (tensor-dims rows) '(1 3))
    (check-equal? (elements rows) '(4.0 5.0 6.0))
    (check-equal? (tensor-dims cols) '(2 2))
    (check-equal? (elements cols) '(2.0 3.0 5.0 6.0))
    (check-false (eq? (tensor-data rows) (tensor-data t)))
    (check-exn #rx"out of range"
               (lambda () (slice-rows t 0 3)))
    (check-exn #rx"out of range"
               (lambda () (slice-cols t 2 1))))

  (test-case "split-last parts the last axis"
    (define t (make-tensor '(2 6) (flvector 1.0 2.0 3.0 4.0 5.0 6.0
                                            7.0 8.0 9.0 10.0 11.0 12.0)))
    (define parts (split-last t 3))
    (check-equal? (length parts) 3)
    (check-equal? (map tensor-dims parts) '((2 2) (2 2) (2 2)))
    (check-equal? (elements (list-ref parts 0)) '(1.0 2.0 7.0 8.0))
    (check-equal? (elements (list-ref parts 1)) '(3.0 4.0 9.0 10.0))
    (check-equal? (elements (list-ref parts 2)) '(5.0 6.0 11.0 12.0))
    (check-equal? (elements (list-ref parts 0))
                  (elements (slice-cols t 0 2)))
    (check-exn #rx"not divisible"
               (lambda () (split-last t 4)))
    (define heads (make-tensor '(2 3 6) (make-flvector 36 1.0)))
    (define split-heads (split-last heads 3))
    (check-equal? (length split-heads) 3)
    (check-equal? (tensor-dims (car split-heads)) '(2 3 2)))

  (test-case "flvector dot product"
    (check-equal? (flvector-dot (flvector 1.0 2.0 3.0)
                                (flvector 4.0 5.0 6.0))
                  32.0)
    (check-equal? (flvector-dot (flvector -1.0 2.0)
                                (flvector 4.0 -3.0))
                  -10.0)
    (check-equal? (flvector-dot (flvector) (flvector)) 0.0)
    (check-exn #rx"lengths differ"
               (lambda ()
                 (flvector-dot (flvector 1.0) (flvector 1.0 2.0))))
    (check-exn exn:fail:contract?
               (lambda ()
                 (flvector-dot '#(1.0) (flvector 1.0)))))
  )