#lang racket/base

(module+ test
  (require json
           racket/flonum
           racket/runtime-path
           rackunit
           "../src/attention.rkt"
           "../src/nn.rkt"
           "../src/tensor.rkt")

  (define (elements t)
    (define data (tensor-data t))
    (for/list ([i (in-range (flvector-length data))])
      (flvector-ref data i)))

  (define (flat-flonums nested)
    (let loop ([nested nested])
      (cond
        [(list? nested) (apply append (map loop nested))]
        [(and (string? nested) (string=? nested "-inf")) (list -inf.0)]
        [(real? nested) (list (real->double-flonum nested))]
        [else (error 'attention-fixture "bad number: ~a" nested)])))

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
      (or (and (eqv? g -inf.0) (eqv? e -inf.0))
          (<= (abs (- g e)) (+ 1e-3 (* 1e-4 (abs e)))))))

  (test-case "q, k, and v are contiguous thirds of the last axis"
    (define mixed (make-tensor '(1 6) (flvector 1.0 2.0 3.0 4.0 5.0 6.0)))
    (define-values (q k v) (qkv-split mixed))
    (check-equal? (elements q) '(1.0 2.0))
    (check-equal? (elements k) '(3.0 4.0))
    (check-equal? (elements v) '(5.0 6.0))
    (define rows (make-tensor '(2 6) (flvector 1.0 2.0 3.0 4.0 5.0 6.0
                                               7.0 8.0 9.0 10.0 11.0 12.0)))
    (define-values (q2 k2 v2) (qkv-split rows))
    (check-equal? (elements q2) '(1.0 2.0 7.0 8.0))
    (check-equal? (elements k2) '(3.0 4.0 9.0 10.0))
    (check-equal? (elements v2) '(5.0 6.0 11.0 12.0)))

  (test-case "a head is a contiguous column block"
    (define q (make-tensor '(2 8)
                           (flvector 0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0
                                     0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0)))
    (define k (make-tensor '(2 8)
                           (flvector 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0
                                     1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)))
    (define v (make-tensor '(2 8)
                           (flvector 8.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0
                                     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)))
    (define head0 (causal-scores (slice-cols q 0 4) (slice-cols k 0 4) 0.5))
    (define head1 (causal-scores (slice-cols q 4 8) (slice-cols k 4 8) 0.5))
    (check-equal? (tensor-ref head0 0 1) -inf.0)
    (check-equal? (tensor-ref head0 1 0) 1.0)
    (check-equal? (tensor-ref head0 1 1) 0.0)
    (check-equal? (tensor-ref head1 1 0) 0.0)
    (define y (attend q k v 2))
    (define p (/ (flexp 1.0) (fl+ (flexp 1.0) 1.0)))
    (check-equal? (tensor-ref y 0 0) 8.0)
    (check-equal? (tensor-ref y 0 4) 3.0)
    (check-= (tensor-ref y 1 0) (* p 8.0) 1e-12)
    (check-equal? (tensor-ref y 1 4) 1.5))

  (test-case "attention scale is 1 over the square root of the head width"
    (check-equal? (attention-scale 64) 0.125)
    (check-equal? (attention-scale 4) 0.5)
    (define q (make-tensor '(1 2) (flvector 2.0 0.0)))
    (define k (make-tensor '(1 2) (flvector 3.0 4.0)))
    (define scores (causal-scores q k (attention-scale 2)))
    (check-= (tensor-ref scores 0 0) (* (flsqrt 0.5) 6.0) 1e-12))

  (test-case "the causal mask is -inf strictly above the diagonal"
    (define q (make-tensor '(2 1) (flvector 1.0 1.0)))
    (define k (make-tensor '(2 1) (flvector 2.0 3.0)))
    (define scores (causal-scores q k 0.5))
    (check-equal? (tensor-ref scores 0 1) -inf.0)
    (check-equal? (tensor-ref scores 0 0) 1.0)
    (check-equal? (tensor-ref scores 1 0) 1.0)
    (check-equal? (tensor-ref scores 1 1) 1.5)
    (define probs (softmax scores))
    (check-equal? (tensor-ref probs 0 1) 0.0)
    (check-equal? (tensor-ref probs 0 0) 1.0))

  (test-case "a cached key prefix is visible to the new query"
    (define q (make-tensor '(1 1) (flvector 1.0)))
    (define k (make-tensor '(3 1) (flvector 1.0 1.0 1.0)))
    (define scores (causal-scores q k 1.0))
    (check-equal? (tensor-dims scores) '(1 3))
    (check-equal? (elements scores) '(1.0 1.0 1.0)))

  (test-case "changing a future token leaves earlier attention outputs unchanged"
    (define n-head 2)
    (define w (make-tensor '(4 12)
                           (flvector 0.1 0.2 -0.1 0.0 0.3 -0.2 0.4 0.1 0.0 -0.3 0.2 0.1
                                     0.0 0.1 0.2 -0.4 0.2 0.0 -0.1 0.3 0.1 0.2 -0.2 0.4
                                     -0.2 0.0 0.1 0.2 0.1 -0.1 0.3 0.0 0.2 0.1 0.4 -0.3
                                     0.2 -0.1 0.0 0.3 -0.2 0.1 0.0 0.2 0.1 -0.4 0.3 0.1)))
    (define b (make-tensor '(12) (flvector 0.01 -0.02 0.03 0.0 0.02 -0.01
                                           0.0 0.01 0.0 -0.03 0.02 0.01)))
    (define pw (make-tensor '(4 4)
                            (flvector 0.2 0.0 -0.1 0.1
                                      0.0 0.3 0.1 -0.2
                                      0.1 0.0 0.2 0.0
                                      -0.1 0.2 0.0 0.4)))
    (define pb (make-tensor '(4) (flvector 0.0 0.1 -0.1 0.05)))
    (define x1 (make-tensor '(3 4)
                            (flvector 1.0 0.0 0.5 -0.5
                                      0.0 1.0 -0.25 0.25
                                      0.5 0.5 0.5 0.5)))
    (define x2 (make-tensor '(3 4)
                            (flvector 1.0 0.0 0.5 -0.5
                                      0.0 1.0 -0.25 0.25
                                      -3.0 4.0 -5.0 6.0)))
    (define y1 (gpt2-attention x1 w b pw pb n-head))
    (define y2 (gpt2-attention x2 w b pw pb n-head))
    (check-equal? (tensor-dims y1) '(3 4))
    (for ([i (in-range 2)])
      (for ([j (in-range 4)])
        (check-equal? (tensor-ref y2 i j) (tensor-ref y1 i j))))
    (check-false (equal? (elements (slice-rows y1 2 3))
                         (elements (slice-rows y2 2 3)))))

  (define-runtime-path fixture-path "attention-cases.json")

  (test-case "one attention layer matches the eager reference"
    (define fixture (call-with-input-file fixture-path read-json))
    (define dims (hash-ref fixture 'dims))
    (define seq (hash-ref dims 'seq))
    (define d (hash-ref dims 'd-model))
    (define n-head (hash-ref dims 'n-head))
    (define x (tensor-from (hash-ref fixture 'x) (list seq d)))
    (define y (gpt2-attention
               x
               (tensor-from (hash-ref fixture 'c-attn-weight) (list d (* 3 d)))
               (tensor-from (hash-ref fixture 'c-attn-bias) (list (* 3 d)))
               (tensor-from (hash-ref fixture 'c-proj-weight) (list d d))
               (tensor-from (hash-ref fixture 'c-proj-bias) (list d))
               n-head))
    (define mixed (matmul x
                          (tensor-from (hash-ref fixture 'c-attn-weight) (list d (* 3 d)))
                          (tensor-from (hash-ref fixture 'c-attn-bias) (list (* 3 d)))))
    (define-values (q k v) (qkv-split mixed))
    (define dh (quotient d n-head))
    (define scale (attention-scale dh))
    (define scores
      (for/list ([h (in-range n-head)])
        (elements (causal-scores (slice-cols q (* h dh) (* (add1 h) dh))
                                 (slice-cols k (* h dh) (* (add1 h) dh))
                                 scale))))
    (define probs (map (lambda (row) (elements (softmax (make-tensor (list seq seq)
                                                                     (flvector-from row)))))
                       scores))
    (define pairs
      (list (list "q" (elements q) (flat-flonums (hash-ref fixture 'q)))
            (list "k" (elements k) (flat-flonums (hash-ref fixture 'k)))
            (list "v" (elements v) (flat-flonums (hash-ref fixture 'v)))
            (list "scores" (apply append scores) (flat-flonums (hash-ref fixture 'scores)))
            (list "probs" (apply append probs) (flat-flonums (hash-ref fixture 'probs)))
            (list "output" (elements y) (flat-flonums (hash-ref fixture 'output)))))
    (for ([pair (in-list pairs)])
      (define name (car pair))
      (define got (cadr pair))
      (define expected (caddr pair))
      (check-equal? (length got) (length expected) name)
      (check-true (within-tolerance? got expected)
                  (format "~a max abs error ~a" name (max-abs-error got expected)))))
  )
