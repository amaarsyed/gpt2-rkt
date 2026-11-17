#lang racket/base

;; Kernel timings use the selected tree's tensors and arithmetic. The
;; tied head is a transposed product, unlike the biased Conv1D cases.

(require json
         racket/cmdline
         racket/flonum
         racket/list
         racket/runtime-path)

(define-runtime-path repo-root "..")

(module+ main
  (define source repo-root)
  (define runs 7)
  (command-line
   #:once-each
   [("--source") path "Source tree to measure" (set! source (path->complete-path path))]
   [("--runs") n "Timed calls after one warmup" (set! runs (string->number n))])
  (unless (exact-positive-integer? runs) (error 'kernels "runs must be positive"))
  (define (binding module name) (dynamic-require (build-path source "src" module) name))
  (define make-tensor (binding "tensor.rkt" 'make-tensor))
  (define tensor-data (binding "tensor.rkt" 'tensor-data))
  (define matmul (binding "nn.rkt" 'matmul))
  (define transposed (binding "nn.rkt" 'matmul-transposed-b))
  (define rows
    (for/list ([shape (in-list '((1 768 2304) (1 768 3072) (1 3072 768)
                                (128 768 3072) (768 768 768) (1 768 50257)))])
      (define n (car shape))
      (define inner (cadr shape))
      (define out (caddr shape))
      (define head? (= out 50257))
      (define x (make-tensor (list n inner) (make-flvector (* n inner) 1.0)))
      (define W (make-tensor (if head? (list out inner) (list inner out))
                             (make-flvector (* inner out) 1.0)))
      (define b (make-tensor (list out) (make-flvector out 0.0)))
      (define (call) (if head? (transposed x W) (matmul x W b)))
      (call)
      (collect-garbage)
      (define samples
        (for/list ([_ (in-range runs)])
          (define start (current-inexact-monotonic-milliseconds))
          (define got (call))
          (define ms (- (current-inexact-monotonic-milliseconds) start))
          (unless (= (flvector-ref (tensor-data got) 0) inner)
            (error 'kernels "wrong matrix result"))
          ms))
      (hasheq 'shape shape 'transposed head? 'samples_ms samples
              'median_ms (list-ref (sort samples <) (quotient runs 2)))))
  (write-json rows)
  (newline))
