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
  (for/fold ([m 0.0]) ([g (in-list got)] [e (in-list expected)])
    (max m (abs (- g e)))))

(define (within-tolerance? got expected
                           #:abs [abs-tol m6-abs-tol]
                           #:rel [rel-tol m6-rel-tol])
  (for/and ([g (in-list got)] [e (in-list expected)])
    (or (and (eqv? g -inf.0) (eqv? e -inf.0))
        (<= (abs (- g e)) (+ abs-tol (* rel-tol (abs e)))))))
