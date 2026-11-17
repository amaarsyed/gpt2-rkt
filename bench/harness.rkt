#lang racket/base

;; Benchmark harness. The prompt-length report is still a placeholder
;; flvector dot product (one "token" is one dot). Each length is measured
;; in its own process because 'peak is a process high-water mark. The
;; matmul report warms up, then reports the median of several runs for
;; the square kernel and for a one-row logits multiply.

(require racket/flonum
         racket/format
         racket/future
         racket/port
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
         matmul-sample-peak-before-gc-bytes
         matmul-sample-rows
         matmul-sample-in
         matmul-sample-out
         run-matmul-benchmark
         format-matmul-report
         default-matmul-dim
         matmul-runs)

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
          "note: peak-bytes is current-memory-use 'peak, the process high-water mark across collections. A later prompt length measured in the same process includes the earlier peak. The harness entry point measures each length in its own process.")
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

(define matmul-runs 5)

(struct matmul-sample (rows in out ms gflops allocated-bytes gc-delta peak-before-gc-bytes)
  #:transparent)

(define (matmul-sample-dim sample)
  (define rows (matmul-sample-rows sample))
  (and (= rows (matmul-sample-in sample) (matmul-sample-out sample))
       rows))

(define (machine-line)
  (define cpu
    (with-handlers ([exn:fail? (lambda (_) "unknown")])
      (call-with-input-file "/proc/cpuinfo"
        (lambda (in)
          (for/or ([line (in-lines in)])
            (and (string-prefix? line "model name")
                 (string-trim (cadr (string-split line ":" #:trim? #f)))))))))
  (format "machine: os=~a arch=~a vm=~a processors=~a cpu=~a"
          (system-type 'os)
          (system-type 'arch)
          (system-type 'vm)
          (processor-count)
          (or cpu "unknown")))

(define (matmul-args rows in out value)
  (values (make-tensor (list rows in) (make-flvector (* rows in) value))
          (make-tensor (list in out) (make-flvector (* in out) value))
          (make-tensor (list out) (make-flvector out 0.0))))

;; 2*rows*in*out flops. Warm up once, collect, then take the median of
;; matmul-runs timed calls. The reported allocation counters are from
;; the median run.
(define (run-matmul-benchmark #:dim [dim #f]
                              #:rows [rows (or dim default-matmul-dim)]
                              #:in [in rows]
                              #:out [out rows]
                              #:runs [runs matmul-runs])
  (unless (exact-positive-integer? rows)
    (raise-argument-error 'run-matmul-benchmark "exact-positive-integer?" rows))
  (unless (exact-positive-integer? in)
    (raise-argument-error 'run-matmul-benchmark "exact-positive-integer?" in))
  (unless (exact-positive-integer? out)
    (raise-argument-error 'run-matmul-benchmark "exact-positive-integer?" out))
  (unless (exact-positive-integer? runs)
    (raise-argument-error 'run-matmul-benchmark "exact-positive-integer?" runs))
  (define-values (x W b) (matmul-args rows in out 1.0))
  (matmul x W b)
  (collect-garbage)
  (define timed
    (for/list ([_ (in-range runs)])
      (define before (make-vector 12))
      (vector-set-performance-stats! before)
      (define alloc-before (current-memory-use 'cumulative))
      (define-values (ms _result)
        (elapsed-ms (lambda () (matmul x W b))))
      (define alloc-after (current-memory-use 'cumulative))
      (define after (make-vector 12))
      (vector-set-performance-stats! after)
      (list ms
            (- alloc-after alloc-before)
            (- (vector-ref after 3) (vector-ref before 3))
            (vector-ref after 11))))
  (define mid (list-ref (sort timed < #:key car) (quotient runs 2)))
  (define ms (car mid))
  (define seconds (/ ms 1000.0))
  (define flops (* 2.0 rows in out))
  (define gflops
    (if (> seconds 0.0)
        (/ flops seconds 1e9)
        +inf.0))
  (matmul-sample rows in out ms gflops (cadr mid) (caddr mid) (cadddr mid)))

(define (format-matmul-report sample)
  (define dim (matmul-sample-dim sample))
  (string-join
   (append
    (list (machine-line)
          (format "matmul-rows: ~a" (matmul-sample-rows sample))
          (format "matmul-in: ~a" (matmul-sample-in sample))
          (format "matmul-out: ~a" (matmul-sample-out sample)))
    (if dim
        (list (format "matmul-dim: ~a" dim))
        '())
    (list (format "matmul-ms: ~a" (format-number (matmul-sample-ms sample)))
          (format "matmul-gflops: ~a" (format-number (matmul-sample-gflops sample)))
          (format "matmul-allocated-bytes: ~a" (matmul-sample-allocated-bytes sample))
          (format "matmul-gc-delta: ~a" (matmul-sample-gc-delta sample))
          (format "matmul-peak-before-gc-bytes: ~a"
                  (matmul-sample-peak-before-gc-bytes sample))
          (format "note: matmul-ms and matmul-gflops are the median of ~a timed calls after one warmup" matmul-runs)
          "note: matmul-allocated-bytes is the current-memory-use 'cumulative delta of the median call"
          "note: matmul-gc-delta is vector-set-performance-stats! slot 3 over that call"
          "note: matmul-peak-before-gc-bytes is vector-set-performance-stats! slot 11, the peak bytes in use just before a garbage collection"))
   "\n"
   #:after-last "\n"))

(define (running-racket)
  (define exec (find-system-path 'exec-file))
  (if (complete-path? exec)
      exec
      (find-executable-path exec)))

(define (run-prompt-process n)
  (define racket (running-racket))
  (define script
    (path->string (simplify-path (path->complete-path (find-system-path 'run-file)))))
  (define-values (proc out in err)
    (subprocess #f #f #f racket script "--prompt-sample" (number->string n)))
  (close-output-port in)
  (define text (port->string out))
  (define err-text (port->string err))
  (close-input-port out)
  (close-input-port err)
  (subprocess-wait proc)
  (unless (zero? (subprocess-status proc))
    (eprintf "~a" err-text)
    (error 'bench "prompt sample ~a failed" n))
  text)

(define (format-prompt-sample sample)
  (string-join
   (list (format "prompt-length: ~a" (bench-sample-prompt-length sample))
         (format "load-ms: ~a" (format-number (bench-sample-load-ms sample)))
         (format "first-token-ms: ~a" (format-number (bench-sample-first-token-ms sample)))
         (format "steady-tokens-per-sec: ~a"
                 (format-number (bench-sample-steady-tokens-per-sec sample)))
         (format "peak-bytes: ~a" (format-number (bench-sample-peak-bytes sample))))
   "\n"
   #:after-last "\n"))

(module+ main
  (define args (vector->list (current-command-line-arguments)))
  (cond
    [(and (= (length args) 2) (string=? (car args) "--prompt-sample"))
     (define n (string->number (cadr args)))
     (unless (exact-positive-integer? n)
       (eprintf "bench: prompt length must be a positive integer\n")
       (exit 1))
     (display (format-prompt-sample (run-sample n default-steady-iters)))]
    [(null? args)
     (display (format-benchmark-report '()))
     (for ([n (in-list (list short-prompt-length long-prompt-length))])
       (display (run-prompt-process n)))
     (display (format-matmul-report (run-matmul-benchmark)))
     (display (format-matmul-report
               (run-matmul-benchmark #:rows 1 #:in 768 #:out 50257)))]
    [else
     (eprintf "bench: unexpected argument: ~a\n" (car args))
     (eprintf "usage: racket bench/harness.rkt\n")
     (exit 1)]))
