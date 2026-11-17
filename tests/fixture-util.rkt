#lang racket/base

;; Shared readers for the eager fixtures. The acceptance bar for a full
;; model is |a-e| <= 1e-3 + 1e-4*|e|. A tiny fixture may pass a tighter
;; absolute bound.

(require racket/flonum
         "../src/tensor.rkt")

(provide m6-abs-tol
         m6-rel-tol
         flat-flonums
         flvector-from
         tensor-from
         max-abs-error
         within-tolerance?)

(define m6-abs-tol 1e-3)
(define m6-rel-tol 1e-4)

(define (flat-flonums nested)
  (let loop ([nested nested])
    (cond
      [(list? nested) (apply append (map loop nested))]
      [(and (string? nested) (string=? nested "-inf")) (list -inf.0)]
      [(real? nested) (list (real->double-flonum nested))]
      [else (error 'fixture "bad number: ~a" nested)])))

(define (flvector-from nums)
  (for/flvector #:length (length nums) ([n (in-list nums)]) n))

(define (tensor-from nested dims)
  (make-tensor dims (flvector-from (flat-flonums nested))))

(define (max-abs-error got expected)
  (unless (= (length got) (length expected))
    (raise-arguments-error 'max-abs-error
                           "lengths differ"
                           "got length" (length got)
                           "expected length" (length expected)))
  (for/fold ([m 0.0]) ([g (in-list got)] [e (in-list expected)])
    (max m (abs (- g e)))))

(define (within-tolerance? got expected
                           #:abs [abs-tol m6-abs-tol]
                           #:rel [rel-tol m6-rel-tol])
  (and (= (length got) (length expected))
       (for/and ([g (in-list got)] [e (in-list expected)])
         (or (and (eqv? g -inf.0) (eqv? e -inf.0))
             (<= (abs (- g e)) (+ abs-tol (* rel-tol (abs e))))))))

(module+ test
  (require rackunit)

  (test-case "a matching prefix is not a matching fixture"
    (for ([got (in-list '(() (1.0) (1.0 2.0 3.0)))])
      (check-false (within-tolerance? got '(1.0 2.0)))
      (check-false (within-tolerance? '(1.0 2.0) got))
      (check-exn #rx"lengths differ" (lambda () (max-abs-error got '(1.0 2.0))))
      (check-exn #rx"lengths differ" (lambda () (max-abs-error '(1.0 2.0) got)))))

  (test-case "equal-length fixtures still use absolute and relative tolerance"
    (check-true (within-tolerance? '(1.0005 100.005 -inf.0) '(1.0 100.0 -inf.0)))
    (check-false (within-tolerance? '(1.1 100.0) '(1.0 100.0)))
    (check-false (within-tolerance? '(+nan.0) '(1.0)))
    (check-= (max-abs-error '(1.0 3.0) '(1.0 2.5)) 0.5 0.0)
    (check-true (within-tolerance? '() '()))
    (check-= (max-abs-error '() '()) 0.0 0.0)))
