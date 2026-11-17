#lang racket/base

(module+ test
  (require json
           racket/flonum
           racket/runtime-path
           rackunit
           "../src/attention.rkt"
           "../src/nn.rkt"
           "../src/tensor.rkt"
           "fixture-util.rkt")

  (define (elements t)
    (define data (tensor-data t))
    (for/list ([i (in-range (flvector-length data))])
      (flvector-ref data i)))

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

  (test-case "the last query row matches that row of the full attend"
    (define nq 4)
    (define nk 6)
    (define d 8)
    (define (fill n salt)
      (define data (make-flvector n))
      (for ([i (in-range n)])
        (flvector-set! data i
                       (real->double-flonum (/ (remainder (+ (* i 17) salt) 11) 5.0))))
      data)
    (define q (make-tensor (list nq d) (fill (* nq d) 1)))
    (define k (make-tensor (list nk d) (fill (* nk d) 2)))
    (define v (make-tensor (list nk d) (fill (* nk d) 3)))
    (define full (attend q k v 2))
    (define one (attend (slice-rows q (sub1 nq) nq) k v 2))
    (check-equal? (elements one) (elements (slice-rows full (sub1 nq) nq)))
    (define big-k (make-tensor (list 10 d) (make-flvector (* 10 d) 99.0)))
    (define big-v (make-tensor (list 10 d) (make-flvector (* 10 d) 99.0)))
    (define kdata (tensor-data k))
    (define vdata (tensor-data v))
    (for ([i (in-range (* nk d))])
      (flvector-set! (tensor-data big-k) i (flvector-ref kdata i))
      (flvector-set! (tensor-data big-v) i (flvector-ref vdata i)))
    (define limited (attend q big-k big-v 2 #:kv-rows nk))
    (check-equal? (elements limited) (elements full))
    (define scale (attention-scale (quotient d 2)))
    (for ([h (in-range 2)])
      (define start (* h (quotient d 2)))
      (define end (+ start (quotient d 2)))
      (define qh (slice-cols q start end))
      (define kh (slice-cols big-k start end #:rows nk))
      (define sliced (causal-scores qh kh scale))
      (define raw (apply-causal-mask! (matmul-transposed-block qh big-k nk start) scale))
      (check-equal? (elements raw) (elements sliced))
      (define vh (slice-cols big-v start end #:rows nk))
      (define bias (make-tensor (list (quotient d 2))
                                (make-flvector (quotient d 2) 0.0)))
      (define probs (softmax sliced))
      (check-equal? (elements (matmul-column-block probs big-v bias start (quotient d 2)))
                    (elements (matmul probs vh bias)))))

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

  (test-case "causal rows match dense attention, including parallel heads and cache capacity"
    (define (filled rows cols salt)
      (define data (make-flvector (* rows cols)))
      (for ([i (in-range (flvector-length data))])
        (flvector-set! data i (/ (real->double-flonum (- (remainder (+ i salt) 17) 8)) 7.0)))
      (make-tensor (list rows cols) data))
    (for ([shape (in-list '((1 7) (4 7) (129 129)))])
      (define nq (car shape))
      (define used (cadr shape))
      (define q (filled nq 16 1))
      (define k (filled (+ used 3) 16 2))
      (define v (filled (+ used 3) 16 3))
      (define expected (make-tensor (list nq 16) (make-flvector (* nq 16) 0.0)))
      (for ([h (in-range 2)])
        (define col (* h 8))
        (define probs (softmax (causal-scores (slice-cols q col (+ col 8))
                                             (slice-cols k col (+ col 8) #:rows used)
                                             (attention-scale 8))))
        (paste-cols! expected 16
                     (matmul probs (slice-cols v col (+ col 8) #:rows used)
                             (make-tensor '(8) (make-flvector 8 0.0))) col))
      (define work (make-attention-work 2 (+ used 3)))
      (define got (attend q k v 2 #:kv-rows used #:workspace work))
      (define saved (elements got))
      (define gap (max-abs-error saved (elements expected)))
      (check-true (< gap 1e-12) (format "dense attention gap: ~a" gap))
      (attend q k v 2 #:kv-rows used #:workspace work)
      (check-equal? (elements got) saved)))

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
      ;; The acceptance bar stays 1e-3 + 1e-4*|e|. This fixture is tighter.
      (check-true (within-tolerance? got expected #:abs 1e-5 #:rel 0.0)
                  (format "~a max abs error ~a" name (max-abs-error got expected)))))
  )
