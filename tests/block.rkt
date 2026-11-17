#lang racket/base

(module+ test
  (require racket/flonum
           rackunit
           "../src/attention.rkt"
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

  (define (flvector-tensor dims n salt)
    (make-tensor dims
                 (flvector-by n (lambda (i)
                                  (real->double-flonum (/ (remainder (+ i salt) 7) 10.0))))))

  (define (flvector-by n proc)
    (define data (make-flvector n))
    (for ([i (in-range n)])
      (flvector-set! data i (proc i)))
    data)

  (define (filled dims value)
    (define n (apply * dims))
    (make-tensor dims (make-flvector n value)))

  (define (params n-head ln1-gain ln1-bias c-attn-w c-attn-b attn-proj-w attn-proj-b
                  ln2-gain ln2-bias c-fc-w c-fc-b mlp-proj-w mlp-proj-b)
    (make-block-params n-head ln1-gain ln1-bias c-attn-w c-attn-b attn-proj-w attn-proj-b
                       ln2-gain ln2-bias c-fc-w c-fc-b mlp-proj-w mlp-proj-b))

  (define (zero-block d n-head ln1-gain ln1-bias ln2-gain ln2-bias
                      c-attn-w attn-proj-w c-fc-w mlp-proj-w)
    (define inner (* 4 d))
    (params n-head
            ln1-gain ln1-bias
            c-attn-w (filled (list (* 3 d)) 0.0)
            attn-proj-w (filled (list d) 0.0)
            ln2-gain ln2-bias
            c-fc-w (filled (list inner) 0.0)
            mlp-proj-w (filled (list d) 0.0)))

  (test-case "the block normalizes with biased variance and eps 1e-5 before attention"
    (check-equal? layernorm-eps 1e-5)
    (define x (make-tensor '(1 2) (flvector 1.0 3.0)))
    (define gain (make-tensor '(2) (flvector 1.0 1.0)))
    (define bias (make-tensor '(2) (flvector 0.0 0.0)))
    (define w (zero-block 2 1 gain bias gain bias
                          (filled '(2 6) 0.0)
                          (filled '(2 2) 0.0)
                          (filled '(2 8) 0.0)
                          (filled '(8 2) 0.0)))
    (define trace (gpt2-block-parts x w))
    (define inv (fl/ 1.0 (flsqrt (fl+ 1.0 layernorm-eps))))
    (check-= (tensor-ref (block-trace-ln1 trace) 0 0) (- inv) 1e-12)
    (check-= (tensor-ref (block-trace-ln1 trace) 0 1) inv 1e-12)
    (define flat (make-tensor '(1 2) (flvector 5.0 5.0)))
    (define flat-bias (make-tensor '(2) (flvector 0.1 -0.2)))
    (define flat-w (zero-block 2 1 gain flat-bias gain bias
                               (filled '(2 6) 0.0)
                               (filled '(2 2) 0.0)
                               (filled '(2 8) 0.0)
                               (filled '(8 2) 0.0)))
    (check-equal? (elements (block-trace-ln1 (gpt2-block-parts flat flat-w)))
                  '(0.1 -0.2)))

  (test-case "pre-LN adds each sublayer onto the residual stream"
    (define x (make-tensor '(2 4) (flvector 1.0 2.0 3.0 4.0
                                            -1.0 0.5 0.25 -0.5)))
    (define w (zero-block 4 2
                          (filled '(4) 0.0) (filled '(4) 0.0)
                          (filled '(4) 0.0) (filled '(4) 0.0)
                          (filled '(4 12) 0.2)
                          (filled '(4 4) 0.3)
                          (filled '(4 16) -0.1)
                          (filled '(16 4) 0.4)))
    (check-equal? (elements (gpt2-block x w)) (elements x))
    (define gain (filled '(4) 1.0))
    (define bias (filled '(4) 0.0))
    (define attn-w (make-tensor '(4 12)
                                (flvector 0.1 0.0 -0.1 0.2 0.0 0.1 0.0 0.2 -0.2 0.1 0.0 0.1
                                          0.0 0.1 0.0 -0.1 0.2 0.0 0.1 0.0 0.2 -0.1 0.1 0.0
                                          -0.1 0.2 0.1 0.0 0.0 0.1 0.2 0.0 -0.1 0.0 0.1 0.2
                                          0.2 -0.1 0.0 0.1 0.1 0.0 0.0 0.1 -0.2 0.2 0.0 0.1)))
    (define attn-b (make-tensor '(12) (flvector 0.01 0.0 -0.01 0.02 0.0 0.0
                                                0.0 0.01 0.0 -0.02 0.01 0.0)))
    (define attn-proj-w (make-tensor '(4 4) (flvector 0.2 0.0 0.1 0.0
                                                      0.0 0.1 0.0 -0.1
                                                      0.1 0.0 0.2 0.0
                                                      0.0 -0.1 0.0 0.2)))
    (define attn-proj-b (filled '(4) 0.0))
    (define fc-w (make-tensor '(4 16)
                              (flvector-by 64 (lambda (i) (real->double-flonum (/ (remainder i 5) 20.0))))))
    (define fc-b (filled '(16) 0.0))
    (define mlp-w (make-tensor '(16 4)
                               (flvector-by 64 (lambda (i) (real->double-flonum (/ (remainder i 3) 25.0))))))
    (define mlp-b (filled '(4) 0.0))
    (define live (params 2 gain bias attn-w attn-b attn-proj-w attn-proj-b
                         gain bias fc-w fc-b mlp-w mlp-b))
    (define ln1 (layer-norm x gain bias))
    (define attn (gpt2-attention ln1 attn-w attn-b attn-proj-w attn-proj-b 2))
    (define res1 (tensor-add x attn))
    (define ln2 (layer-norm res1 gain bias))
    (define hidden (gpt2-mlp-hidden ln2 fc-w fc-b))
    (define mlp (gpt2-mlp ln2 fc-w fc-b mlp-w mlp-b))
    (define expected (tensor-add res1 mlp))
    (define trace (gpt2-block-parts x live))
    (check-equal? (elements (block-trace-ln1 trace)) (elements ln1))
    (check-equal? (elements (block-trace-attn trace)) (elements attn))
    (check-equal? (elements (block-trace-residual1 trace)) (elements res1))
    (check-equal? (elements (block-trace-ln2 trace)) (elements ln2))
    (check-equal? (elements (block-trace-mlp-hidden trace)) (elements hidden))
    (check-equal? (elements (block-trace-out trace)) (elements expected))
    (check-equal? (elements (gpt2-block x live)) (elements expected)))

  (test-case "two blocks then ln_f match the folded pre-LN stack"
    (define x (make-tensor '(2 2) (flvector 0.5 -1.0
                                            1.5 0.25)))
    (define (one salt)
      (define d 2)
      (define inner 8)
      (params 1
              (filled '(2) 1.0)
              (make-tensor '(2) (flvector 0.1 -0.1))
              (flvector-tensor '(2 6) 6 salt)
              (filled '(6) 0.0)
              (flvector-tensor '(2 2) 4 (+ salt 1))
              (filled '(2) 0.0)
              (filled '(2) 1.0)
              (make-tensor '(2) (flvector -0.05 0.05))
              (flvector-tensor '(2 8) 8 (+ salt 2))
              (filled '(8) 0.0)
              (flvector-tensor '(8 2) 8 (+ salt 3))
              (filled '(2) 0.0)))
    (define b1 (one 1))
    (define b2 (one 4))
    (define mid (gpt2-block (gpt2-block x b1) b2))
    (define gain (make-tensor '(2) (flvector 1.5 0.5)))
    (define bias (make-tensor '(2) (flvector 0.2 -0.3)))
    (check-equal? (elements (gpt2-blocks x (list b1 b2))) (elements mid))
    (check-equal? (elements (gpt2-ln-f mid gain bias))
                  (elements (layer-norm mid gain bias)))
    (check-equal? (elements (gpt2-ln-f (gpt2-blocks x (list b1 b2)) gain bias))
                  (elements (layer-norm mid gain bias))))
  )
