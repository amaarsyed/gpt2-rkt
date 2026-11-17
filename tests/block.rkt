#lang racket/base

(module+ test
  (require racket/flonum
           rackunit
           "../src/nn.rkt"
           "../src/tensor.rkt")

  (define (elements t)
    (define data (tensor-data t))
    (for/list ([i (in-range (flvector-length data))])
      (flvector-ref data i)))

  ;; Independent values of the tanh approximation and of the erf GELU.
  (define tanh-gelu-1 0.8411919906082768)
  (define erf-gelu-1 0.8413447460685429)

  (test-case "GELU matches the tanh approximation and not erf"
    (define y (gelu (make-tensor '(3) (flvector 0.0 1.0 2.0))))
    (check-equal? (elements y)
                  (list 0.0 tanh-gelu-1 1.954597694087775))
    (check-true (> (abs (- (tensor-ref y 1) erf-gelu-1)) 1e-4)))
  )
