#lang racket/base

(module+ test
  (require rackunit
           racket/flonum
           racket/string
           "../bench/harness.rkt"
           "../bench/machine.rkt"
           "../src/nn.rkt"
           "../src/tensor.rkt")

  (test-case "flvector dot product"
    (check-equal? (flvector-dot (flvector 1.0 2.0 3.0)
                                (flvector 4.0 5.0 6.0))
                  32.0)
    (check-equal? (flvector-dot (flvector -1.0 2.0)
                                (flvector 4.0 -3.0))
                  -10.0)
    (check-equal? (flvector-dot (flvector) (flvector))
                  0.0))

  (test-case "flvector dot product rejects bad arguments"
    (check-exn exn:fail:contract?
               (lambda ()
                 (flvector-dot (flvector 1.0) (flvector 1.0 2.0))))
    (check-exn exn:fail:contract?
               (lambda ()
                 (flvector-dot '#(1.0 2.0) (flvector 1.0 2.0)))))

  (test-case "placeholder benchmark covers a short and a longer prompt"
    (define iters 10)
    (define n 4)
    (define samples (run-benchmarks #:prompt-lengths (list n) #:steady-iters iters))
    (check-equal? (length samples) 1)
    (define sample (car samples))
    (check-equal? (bench-sample-prompt-length sample) n)
    (check-equal? (bench-sample-dot-sum sample)
                  (* (+ iters 1) n 2.0))
    (check-true (>= (bench-sample-load-ms sample) 0))
    (check-true (>= (bench-sample-first-token-ms sample) 0))
    (check-true (or (= (bench-sample-steady-tokens-per-sec sample) +inf.0)
                    (> (bench-sample-steady-tokens-per-sec sample) 0)))
    (check-true (> (bench-sample-peak-bytes sample) 0)))

  (test-case "default report prints the Racket version, VM, and both lengths"
    (check-true (< short-prompt-length long-prompt-length))
    (define samples (run-benchmarks #:steady-iters 20))
    (check-equal? (map bench-sample-prompt-length samples)
                  (list short-prompt-length long-prompt-length))
    (define text (format-benchmark-report samples))
    (check-regexp-match (regexp-quote (format "racket-version: ~a" (version))) text)
    (check-regexp-match (regexp-quote (format "vm: ~a" (system-type 'vm))) text)
    (check-regexp-match #px"workload: flvector-dot" text)
    (check-regexp-match (regexp-quote (format "prompt-length: ~a" short-prompt-length)) text)
    (check-regexp-match (regexp-quote (format "prompt-length: ~a" long-prompt-length)) text)
    (check-regexp-match #px"load-ms: " text)
    (check-regexp-match #px"first-token-ms: " text)
    (check-regexp-match #px"steady-tokens-per-sec: " text)
    (check-regexp-match #px"peak-bytes: " text)
    (check-true (string-contains? text "placeholder kernel"))
    (check-true (string-contains? text "current-memory-use 'peak"))
    (check-true (string-contains? text "later prompt length")))

  (test-case "matmul benchmark times the square kernel"
    (define sample (run-matmul-benchmark #:dim 8))
    (check-equal? (matmul-sample-dim sample) 8)
    (check-true (>= (matmul-sample-ms sample) 0))
    (check-true (or (= (matmul-sample-gflops sample) +inf.0)
                    (> (matmul-sample-gflops sample) 0)))
    (check-true (>= (matmul-sample-allocated-bytes sample) 0))
    (check-true (>= (matmul-sample-gc-delta sample) 0))
    (define text (format-matmul-report sample))
    (check-regexp-match #px"matmul-dim: 8" text)
    (check-regexp-match #px"matmul-gflops: " text)
    (check-regexp-match #px"matmul-allocated-bytes: " text)
    (check-regexp-match #px"matmul-gc-delta: " text)
    (check-regexp-match #px"matmul-peak-before-gc-bytes: " text)
    (check-regexp-match #px"median of 5" text)
    (check-regexp-match #px"machine: os=" text)
    (check-regexp-match #px"physical-cpu=" text)
    (check-regexp-match #px"ncpu=" text)
    (define-values (x W b)
      (let ([data (make-flvector 64 1.0)])
        (values (make-tensor '(8 8) data)
                (make-tensor '(8 8) (make-flvector 64 1.0))
                (make-tensor '(8) (make-flvector 8 0.0)))))
    (check-equal? (tensor-ref (matmul x W b) 3 4) 8.0))

  (test-case "the CPU name is the Linux model or the macOS brand string"
    (check-equal? (cpu-name-from "Intel(R) Xeon(R) Processor" "Apple M5 Pro")
                  "Intel(R) Xeon(R) Processor")
    (check-equal? (cpu-name-from #f "Apple M5 Pro") "Apple M5 Pro")
    (check-equal? (cpu-name-from "" "  Apple M5 Pro  ") "Apple M5 Pro")
    (check-equal? (cpu-name-from "   " "") #f)
    (define live (host-cpu-name))
    (check-true (non-empty-string? live))
    (check-false (string=? live "unknown"))))
