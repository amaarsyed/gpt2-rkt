#lang racket/base

;; Benchmark harness. The prompt-length report is still a placeholder
;; flvector dot product (one "token" is one dot). The matmul report times
;; the Conv1D kernel on a square problem, default 768×768.

(require racket/flonum
         racket/format
         racket/string
         "../src/nn.rkt"
         "../src/tensor.rkt")

(provide flvector-dot
         short-prompt-length
         long-prompt-length
         bench-sample?
         bench-sample-prompt-length
         bench-sample-load-ms
         bench-sample-first-token-ms
         bench-sample-steady-tokens-per-sec
         bench-sample-peak-bytes
         bench-sample-dot-sum
         run-benchmarks
         format-benchmark-report
         matmul-sample?
         matmul-sample-dim
         matmul-sample-ms
         matmul-sample-gflops
         matmul-sample-allocated-bytes
         matmul-sample-gc-delta
         matmul-sample-stats-slot-11
         run-matmul-benchmark
         format-matmul-report
         default-matmul-dim)

(define short-prompt-length 16)
(define long-prompt-length 128)
(define default-steady-iters 2000)

(struct bench-sample (prompt-length
                      load-ms
                      first-token-ms
                      steady-tokens-per-sec
                      peak-bytes
                      dot-sum)
  #:transparent)

(define (make-filled-flvector n value)
  (make-flvector n value))

(define (elapsed-ms thunk)
  (define start (current-inexact-monotonic-milliseconds))
  (define result (thunk))
  (values (- (current-inexact-monotonic-milliseconds) start) result))

(define (run-sample prompt-length steady-iters)
  (unless (exact-positive-integer? prompt-length)
    (raise-argument-error 'run-benchmarks "exact-positive-integer?" prompt-length))
  (unless (exact-positive-integer? steady-iters)
    (raise-argument-error 'run-benchmarks "exact-positive-integer?" steady-iters))
  (define-values (load-ms loaded)
    (elapsed-ms
     (lambda ()
       (cons (make-filled-flvector prompt-length 1.0)
             (make-filled-flvector prompt-length 2.0)))))
  (define left (car loaded))
  (define right (cdr loaded))
  (define-values (first-ms first-dot)
    (elapsed-ms (lambda () (flvector-dot left right))))
  (define-values (steady-ms steady-sum)
    (elapsed-ms
     (lambda ()
       (let loop ([i 0] [acc 0.0])
         (cond
           [(= i steady-iters) acc]
           [else
            (loop (add1 i) (fl+ acc (flvector-dot left right)))])))))
  (define steady-sec (/ steady-ms 1000.0))
  (define tokens-per-sec
    (if (> steady-sec 0.0)
        (/ (real->double-flonum steady-iters) steady-sec)
        +inf.0))
  (bench-sample prompt-length
                load-ms
                first-ms
                tokens-per-sec
                (current-memory-use 'peak)
                (fl+ first-dot steady-sum)))

(define (run-benchmarks #:prompt-lengths [prompt-lengths (list short-prompt-length
                                                               long-prompt-length)]
                        #:steady-iters [steady-iters default-steady-iters])
  (unless (and (pair? prompt-lengths) (list? prompt-lengths))
    (raise-argument-error 'run-benchmarks
                          "(non-empty-listof exact-positive-integer?)"
                          prompt-lengths))
  (for/list ([n (in-list prompt-lengths)])
    (run-sample n steady-iters)))

(define (format-number n)
  (cond
    [(exact-integer? n) (number->string n)]
    [(rational? n) (~r n #:precision '(= 3))]
    [else (number->string n)]))

(define (format-benchmark-report samples)
  (string-join
   (append
    (list (format "racket-version: ~a" (version))
          (format "vm: ~a" (system-type 'vm))
          "workload: flvector-dot"
          "note: placeholder kernel until model load exists; one token is one dot product"
          "note: peak-bytes is current-memory-use 'peak, the process high-water mark across collections")
    (apply append
           (for/list ([sample (in-list samples)])
             (list
              (format "prompt-length: ~a" (bench-sample-prompt-length sample))
              (format "load-ms: ~a" (format-number (bench-sample-load-ms sample)))
              (format "first-token-ms: ~a" (format-number (bench-sample-first-token-ms sample)))
              (format "steady-tokens-per-sec: ~a"
                      (format-number (bench-sample-steady-tokens-per-sec sample)))
              (format "peak-bytes: ~a" (format-number (bench-sample-peak-bytes sample)))))))
   "\n"
   #:after-last "\n"))

(define default-matmul-dim 768)

(struct matmul-sample (dim ms gflops allocated-bytes gc-delta stats-slot-11)
  #:transparent)

(define (square-matmul-args dim value)
  (define n (* dim dim))
  (define data (make-flvector n value))
  (define bias (make-flvector dim 0.0))
  (values (make-tensor (list dim dim) data)
          (make-tensor (list dim dim) (make-flvector n value))
          (make-tensor (list dim) bias)))

;; 2*dim^3 flops. Warm up once, collect, then time a second call.
(define (run-matmul-benchmark #:dim [dim default-matmul-dim])
  (unless (exact-positive-integer? dim)
    (raise-argument-error 'run-matmul-benchmark "exact-positive-integer?" dim))
  (define-values (x W b) (square-matmul-args dim 1.0))
  (matmul x W b)
  (collect-garbage)
  (define before (make-vector 12))
  (vector-set-performance-stats! before)
  (define alloc-before (current-memory-use 'cumulative))
  (define-values (ms _result)
    (elapsed-ms (lambda () (matmul x W b))))
  (define alloc-after (current-memory-use 'cumulative))
  (define after (make-vector 12))
  (vector-set-performance-stats! after)
  (define seconds (/ ms 1000.0))
  (define flops (* 2.0 dim dim dim))
  (define gflops
    (if (> seconds 0.0)
        (/ flops seconds 1e9)
        +inf.0))
  (matmul-sample dim
                 ms
                 gflops
                 (- alloc-after alloc-before)
                 (- (vector-ref after 3) (vector-ref before 3))
                 (vector-ref after 11)))

(define (format-matmul-report sample)
  (string-join
   (list (format "matmul-dim: ~a" (matmul-sample-dim sample))
         (format "matmul-ms: ~a" (format-number (matmul-sample-ms sample)))
         (format "matmul-gflops: ~a" (format-number (matmul-sample-gflops sample)))
         (format "matmul-allocated-bytes: ~a" (matmul-sample-allocated-bytes sample))
         (format "matmul-gc-delta: ~a" (matmul-sample-gc-delta sample))
         (format "matmul-stats-slot-11: ~a" (matmul-sample-stats-slot-11 sample))
         "note: matmul-allocated-bytes is the current-memory-use 'cumulative delta of the timed call"
         "note: matmul-gc-delta is vector-set-performance-stats! slot 3 over that call"
         "note: matmul-stats-slot-11 is vector-set-performance-stats! slot 11 after that call")
   "\n"
   #:after-last "\n"))

(module+ main
  (define args (vector->list (current-command-line-arguments)))
  (unless (null? args)
    (eprintf "bench: unexpected argument: ~a\n" (car args))
    (eprintf "usage: racket bench/harness.rkt\n")
    (exit 1))
  (display (format-benchmark-report (run-benchmarks)))
  (display (format-matmul-report (run-matmul-benchmark))))
