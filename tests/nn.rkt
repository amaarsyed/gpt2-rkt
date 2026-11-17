#lang racket/base

(module+ test
  (require rackunit
           racket/flonum
           "../src/nn.rkt"
           "../src/tensor.rkt")

  (define (elements t)
    (define data (tensor-data t))
    (for/list ([i (in-range (flvector-length data))])
      (flvector-ref data i)))

  (define (check-close t expected eps)
    (define got (elements t))
    (check-equal? (length got) (length expected))
    (for ([g (in-list got)] [e (in-list expected)])
      (check-= g e eps)))

  (test-case "matmul is x[n, in] times W[in, out] plus bias"
    (define x (make-tensor '(2 2) (flvector 1.0 2.0 3.0 4.0)))
    (define W (make-tensor '(2 3) (flvector 5.0 6.0 7.0 8.0 9.0 10.0)))
    (define b (make-tensor '(3) (flvector 1.0 0.0 -1.0)))
    (define y (matmul x W b))
    (check-equal? (tensor-dims y) '(2 3))
    (check-equal? (elements y) '(22.0 24.0 26.0 48.0 54.0 60.0)))

  (test-case "matmul broadcasts bias when the inner dimension is empty"
    (define x (make-tensor '(2 0) (flvector)))
    (define W (make-tensor '(0 3) (flvector)))
    (define b (make-tensor '(3) (flvector 1.0 2.0 3.0)))
    (check-equal? (elements (matmul x W b)) '(1.0 2.0 3.0 1.0 2.0 3.0))
    (define empty-rows
      (matmul (make-tensor '(0 3) (flvector))
              (make-tensor '(3 2) (flvector 1.0 2.0 3.0 4.0 5.0 6.0))
              (make-tensor '(2) (flvector 9.0 8.0))))
    (check-equal? (tensor-dims empty-rows) '(0 2))
    (check-equal? (elements empty-rows) '()))

  (test-case "matmul reports shapes when the contraction does not match"
    (define x (make-tensor '(2 3) (flvector 1.0 2.0 3.0 4.0 5.0 6.0)))
    (define W (make-tensor '(2 2) (flvector 1.0 0.0 0.0 1.0)))
    (define b (make-tensor '(2) (flvector 0.0 0.0)))
    (check-exn #rx"\\(2 3\\)" (lambda () (matmul x W b)))
    (check-exn #rx"\\(2 2\\)" (lambda () (matmul x W b)))
    (check-exn #rx"bias"
               (lambda ()
                 (matmul (make-tensor '(1 2) (flvector 1.0 2.0))
                         W
                         (make-tensor '(3) (flvector 1.0 2.0 3.0)))))
    (check-exn exn:fail:contract?
               (lambda ()
                 (matmul (make-tensor '(2 2 1) (flvector 1.0 2.0 3.0 4.0))
                         W
                         b))))

  (test-case "transposed-b dots h[n, in] with W stored [out, in]"
    (define W (make-tensor '(2 3) (flvector 4.0 5.0 6.0 7.0 8.0 9.0)))
    (check-equal? (elements (matmul-transposed-b
                             (make-tensor '(1 3) (flvector 1.0 2.0 3.0))
                             W))
                  '(32.0 50.0))
    (check-equal? (elements (matmul-transposed-b
                             (make-tensor '(1 3) (flvector 1.0 0.0 -1.0))
                             W))
                  '(-2.0 -2.0))
    (check-exn #rx"\\(1 2\\)"
               (lambda ()
                 (matmul-transposed-b (make-tensor '(1 2) (flvector 1.0 2.0)) W))))

  (test-case "parallel matmul matches the integer formula"
    (define dim 64)
    (define x-data (make-flvector (* dim dim)))
    (define w-data (make-flvector (* dim dim)))
    (for ([i (in-range dim)])
      (for ([k (in-range dim)])
        (flvector-set! x-data (+ (* i dim) k) (exact->inexact (+ i 1)))))
    (for ([k (in-range dim)])
      (for ([j (in-range dim)])
        (flvector-set! w-data (+ (* k dim) j) (exact->inexact (+ j 1)))))
    (define y (matmul (make-tensor (list dim dim) x-data)
                      (make-tensor (list dim dim) w-data)
                      (make-tensor (list dim) (make-flvector dim 0.0))))
    (check-equal? (tensor-dims y) (list dim dim))
    (for ([i (in-range dim)])
      (for ([j (in-range dim)])
        (check-equal? (tensor-ref y i j)
                      (exact->inexact (* (+ i 1) (+ j 1) dim))))))

  (test-case "softmax subtracts the row max and is zero on an all-masked row"
    (define plain (make-tensor '(3) (flvector 0.0 1.0 -1.0)))
    (define shifted (make-tensor '(3) (flvector 1000.0 1001.0 999.0)))
    (define expected '(0.24472847105479764 0.6652409557748218 0.09003057317038046))
    (check-close (softmax plain) expected 1e-15)
    (check-close (softmax shifted) expected 1e-15)
    (check-equal? (elements (softmax plain)) (elements (softmax shifted)))
    (check-equal? (elements (softmax (make-tensor '(3) (flvector -inf.0 4.0 -inf.0))))
                  '(0.0 1.0 0.0))
    (check-equal? (elements (softmax (make-tensor '(3) (flvector -inf.0 -inf.0 -inf.0))))
                  '(0.0 0.0 0.0))
    (define rows (softmax (make-tensor '(2 3) (flvector 0.0 1.0 -1.0
                                                      -inf.0 -inf.0 -inf.0))))
    (check-close rows (append expected '(0.0 0.0 0.0)) 1e-15)
    (check-exn #rx"positive"
               (lambda ()
                 (softmax (make-tensor '(2 0) (flvector))))))

  (test-case "layer-norm uses biased variance and eps 1e-5"
    (check-equal? layernorm-eps 1e-5)
    (define y (layer-norm (make-tensor '(3) (flvector 1.0 2.0 3.0))
                          (make-tensor '(3) (flvector 1.0 1.0 1.0))
                          (make-tensor '(3) (flvector 0.0 0.0 0.0))))
    (check-close y '(-1.2247356859083902 0.0 1.2247356859083902) 1e-15)
    (define scaled (layer-norm (make-tensor '(2) (flvector 1.0 3.0))
                               (make-tensor '(2) (flvector 2.0 0.5))
                               (make-tensor '(2) (flvector 0.1 -0.2))))
    (check-close scaled '(-1.8999900000749994 0.29999750001874986) 1e-15)
    (define flat (layer-norm (make-tensor '(1 3) (flvector 5.0 5.0 5.0))
                             (make-tensor '(3) (flvector 2.0 3.0 4.0))
                             (make-tensor '(3) (flvector 0.1 0.2 0.3))))
    (check-equal? (elements flat) '(0.1 0.2 0.3))
    (check-exn #rx"gain"
               (lambda ()
                 (layer-norm (make-tensor '(2) (flvector 1.0 2.0))
                             (make-tensor '(1) (flvector 1.0))
                             (make-tensor '(2) (flvector 0.0 0.0))))))

  (test-case "layer-norm of a constant row is the bias"
    (define flat (layer-norm (make-tensor '(4) (flvector 7.0 7.0 7.0 7.0))
                             (make-tensor '(4) (flvector 9.0 8.0 7.0 6.0))
                             (make-tensor '(4) (flvector -1.0 0.5 2.0 3.0))))
    (check-equal? (elements flat) '(-1.0 0.5 2.0 3.0)))

  (test-case "gelu matches the tanh approximation at fixed inputs"
    (define y (gelu (make-tensor '(2 3) (flvector 0.0 1.0 -1.0
                                                2.0 -2.0 0.5))))
    (check-equal? (tensor-dims y) '(2 3))
    (check-close y '(0.0
                     0.8411919906082768
                     -0.15880800939172324
                     1.954597694087775
                     -0.04540230591222494
                     0.34571400982514394)
                 1e-15))
  )