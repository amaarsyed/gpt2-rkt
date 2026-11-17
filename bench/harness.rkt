#lang racket/base

;; Benchmark harness. Until generation exists, the workload is an flvector
;; dot product: prompt length is the vector length, and one "token" is one
;; dot product. The report shape (load, first token, steady tokens/sec,
;; peak memory, short and longer prompt) is what model runs will fill in.

(require racket/flonum
         racket/format
         racket/string)

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
         format-benchmark-report)

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

(define (flvector-dot a b)
  (unless (flvector? a)
    (raise-argument-error 'flvector-dot "flvector?" 0 a b))
  (unless (flvector? b)
    (raise-argument-error 'flvector-dot "flvector?" 1 a b))
  (define n (flvector-length a))
  (unless (= n (flvector-length b))
    (raise-arguments-error 'flvector-dot
                           "lengths differ"
                           "first length" n
                           "second length" (flvector-length b)))
  (let loop ([i 0] [acc 0.0])
    (cond
      [(= i n) acc]
      [else
       (loop (add1 i)
             (fl+ acc (fl* (flvector-ref a i) (flvector-ref b i))))])))

(define (fill-flvector! vec value)
  (define n (flvector-length vec))
  (let loop ([i 0])
    (unless (= i n)
      (flvector-set! vec i value)
      (loop (add1 i)))))

(define (make-filled-flvector n value)
  (define vec (make-flvector n))
  (fill-flvector! vec value)
  vec)

(define (elapsed-ms thunk)
  (define start (current-inexact-monotonic-milliseconds))
  (define result (thunk))
  (values (- (current-inexact-monotonic-milliseconds) start) result))

(define (run-sample prompt-length steady-iters)
  (unless (exact-positive-integer? prompt-length)
    (raise-argument-error 'run-benchmarks "exact-positive-integer?" prompt-length))
  (unless (exact-positive-integer? steady-iters)
    (raise-argument-error 'run-benchmarks "exact-positive-integer?" steady-iters))
  (define peak (current-memory-use))
  (define (note!)
    (set! peak (max peak (current-memory-use))))
  (define-values (load-ms loaded)
    (elapsed-ms
     (lambda ()
       (cons (make-filled-flvector prompt-length 1.0)
             (make-filled-flvector prompt-length 2.0)))))
  (note!)
  (define left (car loaded))
  (define right (cdr loaded))
  (define-values (first-ms first-dot)
    (elapsed-ms (lambda () (flvector-dot left right))))
  (note!)
  (define-values (steady-ms steady-sum)
    (elapsed-ms
     (lambda ()
       (let loop ([i 0] [acc 0.0])
         (cond
           [(= i steady-iters) acc]
           [else
            (loop (add1 i) (fl+ acc (flvector-dot left right)))])))))
  (note!)
  (define steady-sec (/ steady-ms 1000.0))
  (define tokens-per-sec
    (if (> steady-sec 0.0)
        (/ (real->double-flonum steady-iters) steady-sec)
        +inf.0))
  (bench-sample prompt-length
                load-ms
                first-ms
                tokens-per-sec
                peak
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
          "note: peak-bytes is the max current-memory-use sample, not OS RSS")
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

(module+ main
  (define args (vector->list (current-command-line-arguments)))
  (unless (null? args)
    (eprintf "bench: unexpected argument: ~a\n" (car args))
    (eprintf "usage: racket bench/harness.rkt\n")
    (exit 1))
  (display (format-benchmark-report (run-benchmarks))))
