#lang racket/base

(module+ test
  (require json
           racket/flonum
           racket/runtime-path
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

  (define (flvector-tensor dims salt)
    (define n (apply * dims))
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

  (define (zero-block d n-head ln1-gain ln1-bias ln2-gain ln2-bias
                      c-attn-w attn-proj-w c-fc-w mlp-proj-w)
    (define inner (* 4 d))
    (make-block-params #:n-head n-head
                       #:ln1-gain ln1-gain
                       #:ln1-bias ln1-bias
                       #:c-attn-w c-attn-w
                       #:c-attn-b (filled (list (* 3 d)) 0.0)
                       #:attn-proj-w attn-proj-w
                       #:attn-proj-b (filled (list d) 0.0)
                       #:ln2-gain ln2-gain
                       #:ln2-bias ln2-bias
                       #:c-fc-w c-fc-w
                       #:c-fc-b (filled (list inner) 0.0)
                       #:mlp-proj-w mlp-proj-w
                       #:mlp-proj-b (filled (list d) 0.0)))

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
    (define live (make-block-params #:n-head 2
                                     #:ln1-gain gain
                                     #:ln1-bias bias
                                     #:c-attn-w attn-w
                                     #:c-attn-b attn-b
                                     #:attn-proj-w attn-proj-w
                                     #:attn-proj-b attn-proj-b
                                     #:ln2-gain gain
                                     #:ln2-bias bias
                                     #:c-fc-w fc-w
                                     #:c-fc-b fc-b
                                     #:mlp-proj-w mlp-w
                                     #:mlp-proj-b mlp-b))
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

  (test-case "block tensors are keyed by the Hugging Face names"
    (define tensors
      (hash "ln_1.weight" (filled '(2) 1.0)
            "ln_1.bias" (filled '(2) 0.2)
            "attn.c_attn.weight" (filled '(2 6) 0.3)
            "attn.c_attn.bias" (filled '(6) 0.4)
            "attn.c_proj.weight" (filled '(2 2) 0.5)
            "attn.c_proj.bias" (filled '(2) 0.6)
            "ln_2.weight" (filled '(2) 0.7)
            "ln_2.bias" (filled '(2) 0.8)
            "mlp.c_fc.weight" (filled '(2 8) 0.9)
            "mlp.c_fc.bias" (filled '(8) 1.0)
            "mlp.c_proj.weight" (filled '(8 2) 1.1)
            "mlp.c_proj.bias" (filled '(2) 1.2)))
    (define got (block-params-from-tensors #:n-head 2 tensors))
    (define expected
      (make-block-params #:n-head 2
                         #:ln1-gain (hash-ref tensors "ln_1.weight")
                         #:ln1-bias (hash-ref tensors "ln_1.bias")
                         #:c-attn-w (hash-ref tensors "attn.c_attn.weight")
                         #:c-attn-b (hash-ref tensors "attn.c_attn.bias")
                         #:attn-proj-w (hash-ref tensors "attn.c_proj.weight")
                         #:attn-proj-b (hash-ref tensors "attn.c_proj.bias")
                         #:ln2-gain (hash-ref tensors "ln_2.weight")
                         #:ln2-bias (hash-ref tensors "ln_2.bias")
                         #:c-fc-w (hash-ref tensors "mlp.c_fc.weight")
                         #:c-fc-b (hash-ref tensors "mlp.c_fc.bias")
                         #:mlp-proj-w (hash-ref tensors "mlp.c_proj.weight")
                         #:mlp-proj-b (hash-ref tensors "mlp.c_proj.bias")))
    (check-equal? got expected)
    (check-exn #rx"missing tensor"
               (lambda () (block-params-from-tensors #:n-head 1 (hash)))))

  (test-case "two blocks then ln_f match the folded pre-LN stack"
    (define x (make-tensor '(2 2) (flvector 0.5 -1.0
                                            1.5 0.25)))
    (define (one salt)
      (define d 2)
      (define inner 8)
      (make-block-params #:n-head 1
                         #:ln1-gain (filled '(2) 1.0)
                         #:ln1-bias (make-tensor '(2) (flvector 0.1 -0.1))
                         #:c-attn-w (flvector-tensor '(2 6) salt)
                         #:c-attn-b (filled '(6) 0.0)
                         #:attn-proj-w (flvector-tensor '(2 2) (+ salt 1))
                         #:attn-proj-b (filled '(2) 0.0)
                         #:ln2-gain (filled '(2) 1.0)
                         #:ln2-bias (make-tensor '(2) (flvector -0.05 0.05))
                         #:c-fc-w (flvector-tensor '(2 8) (+ salt 2))
                         #:c-fc-b (filled '(8) 0.0)
                         #:mlp-proj-w (flvector-tensor '(8 2) (+ salt 3))
                         #:mlp-proj-b (filled '(2) 0.0)))
    (define b1 (one 1))
    (define b2 (one 4))
    (define mid (gpt2-block (gpt2-block x b1) b2))
    (define gain (make-tensor '(2) (flvector 1.5 0.5)))
    (define bias (make-tensor '(2) (flvector 0.2 -0.3)))
    (check-equal? (elements (gpt2-blocks x (list b1 b2))) (elements mid))
    (check-equal? (elements (layer-norm (gpt2-blocks x (list b1 b2)) gain bias))
                  (elements (layer-norm mid gain bias))))

  (define (flat-flonums nested)
    (let loop ([nested nested])
      (cond
        [(list? nested) (apply append (map loop nested))]
        [(and (string? nested) (string=? nested "-inf")) (list -inf.0)]
        [(real? nested) (list (real->double-flonum nested))]
        [else (error 'block-fixture "bad number: ~a" nested)])))

  (define (flvector-from nums)
    (define data (make-flvector (length nums)))
    (for ([n (in-list nums)] [i (in-naturals)])
      (flvector-set! data i n))
    data)

  (define (tensor-from nested dims)
    (make-tensor dims (flvector-from (flat-flonums nested))))

  (define (max-abs-error got expected)
    (for/fold ([m 0.0]) ([g (in-list got)] [e (in-list expected)])
      (max m (abs (- g e)))))

  (define (within-tolerance? got expected)
    (for/and ([g (in-list got)] [e (in-list expected)])
      (<= (abs (- g e)) (+ 1e-3 (* 1e-4 (abs e))))))

  (define-runtime-path fixture-path "block-cases.json")

  (define (params-from block d inner n-head)
    (make-block-params #:n-head n-head
                       #:ln1-gain (tensor-from (hash-ref block 'ln1-gain) (list d))
                       #:ln1-bias (tensor-from (hash-ref block 'ln1-bias) (list d))
                       #:c-attn-w (tensor-from (hash-ref block 'c-attn-weight) (list d (* 3 d)))
                       #:c-attn-b (tensor-from (hash-ref block 'c-attn-bias) (list (* 3 d)))
                       #:attn-proj-w (tensor-from (hash-ref block 'attn-proj-weight) (list d d))
                       #:attn-proj-b (tensor-from (hash-ref block 'attn-proj-bias) (list d))
                       #:ln2-gain (tensor-from (hash-ref block 'ln2-gain) (list d))
                       #:ln2-bias (tensor-from (hash-ref block 'ln2-bias) (list d))
                       #:c-fc-w (tensor-from (hash-ref block 'c-fc-weight) (list d inner))
                       #:c-fc-b (tensor-from (hash-ref block 'c-fc-bias) (list inner))
                       #:mlp-proj-w (tensor-from (hash-ref block 'mlp-proj-weight) (list inner d))
                       #:mlp-proj-b (tensor-from (hash-ref block 'mlp-proj-bias) (list d))))

  (test-case "two eager blocks and ln_f match within tolerance"
    (define fixture (call-with-input-file fixture-path read-json))
    (define dims (hash-ref fixture 'dims))
    (define seq (hash-ref dims 'seq))
    (define d (hash-ref dims 'd-model))
    (define inner (hash-ref dims 'inner))
    (define n-head (hash-ref dims 'n-head))
    (define blocks (hash-ref fixture 'blocks))
    (define hidden (tensor-from (hash-ref fixture 'x) (list seq d)))
    (define params-list
      (for/list ([block (in-list blocks)])
        (params-from block d inner n-head)))
    (for ([block (in-list blocks)] [p (in-list params-list)])
      (define trace (gpt2-block-parts hidden p))
      (define pairs
        (list (list "ln1" (elements (block-trace-ln1 trace)) (flat-flonums (hash-ref block 'ln1)))
              (list "attn" (elements (block-trace-attn trace)) (flat-flonums (hash-ref block 'attn)))
              (list "residual1" (elements (block-trace-residual1 trace)) (flat-flonums (hash-ref block 'residual1)))
              (list "ln2" (elements (block-trace-ln2 trace)) (flat-flonums (hash-ref block 'ln2)))
              (list "mlp-hidden" (elements (block-trace-mlp-hidden trace)) (flat-flonums (hash-ref block 'mlp-hidden)))
              (list "out" (elements (block-trace-out trace)) (flat-flonums (hash-ref block 'out)))))
      (for ([pair (in-list pairs)])
        (check-equal? (length (cadr pair)) (length (caddr pair)) (car pair))
        (check-true (within-tolerance? (cadr pair) (caddr pair))
                    (format "~a max abs error ~a" (car pair) (max-abs-error (cadr pair) (caddr pair)))))
      (set! hidden (block-trace-out trace)))
    (define stacked (gpt2-blocks (tensor-from (hash-ref fixture 'x) (list seq d)) params-list))
    (define final (layer-norm stacked
                             (tensor-from (hash-ref fixture 'ln-f-gain) (list d))
                             (tensor-from (hash-ref fixture 'ln-f-bias) (list d))))
    (define expected-final (flat-flonums (hash-ref fixture 'ln-f)))
    (check-true (within-tolerance? (elements final) expected-final)
                (format "ln_f max abs error ~a" (max-abs-error (elements final) expected-final))))
  )
