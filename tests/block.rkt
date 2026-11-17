#lang racket/base

(module+ test
  (require racket/flonum
           rackunit
           "../src/block.rkt"
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

  (test-case "c_fc maps model width to the inner width and c_proj maps back"
    (define x (make-tensor '(1 2) (flvector 1.0 0.0)))
    (define c-fc-w (make-tensor '(2 4) (flvector 1.0 0.0 0.0 0.0
                                                 0.0 0.0 0.0 0.0)))
    (define c-fc-b (make-tensor '(4) (flvector 0.0 0.0 0.0 0.0)))
    (define c-proj-w (make-tensor '(4 2) (flvector 1.0 0.0
                                                   0.0 0.0
                                                   0.0 0.0
                                                   0.0 0.0)))
    (define c-proj-b (make-tensor '(2) (flvector 0.0 0.0)))
    (define y (gpt2-mlp x c-fc-w c-fc-b c-proj-w c-proj-b))
    (check-equal? (tensor-dims y) '(1 2))
    (check-= (tensor-ref y 0 0) tanh-gelu-1 1e-15)
    (check-equal? (tensor-ref y 0 1) 0.0)
    (check-exn exn:fail:contract?
               (lambda ()
                 (gpt2-mlp x
                           (make-tensor '(4 2) (flvector 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0))
                           c-fc-b
                           c-proj-w
                           c-proj-b))))
  )
