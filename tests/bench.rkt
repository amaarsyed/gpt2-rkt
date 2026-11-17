#lang racket/base

(module+ test
  (require racket/runtime-path
           racket/string
           rackunit
           "../bench/machine.rkt"
           "../bench/run.rkt"
           "../src/model.rkt")

  (define-runtime-path tiny-dir "tiny-gpt2")

  (define (row kind . pairs)
    (apply hasheq 'kind kind 'pid 1 pairs))

  (test-case "a repeated prompt has the requested length"
    (check-equal? (ids-of-length '(11 262) 5) '(11 262 11 262 11))
    (check-equal? (length (ids-of-length '(1 2 3 4) 16)) 16)
    (check-equal? prefill-lengths '(16 128 512 1000)))

  (test-case "the median is the middle of the sorted runs"
    (check-equal? (median-number '(3 1 2)) 2)
    (check-equal? (median-number '(4 1 3 2)) 3))

  (test-case "greedy ids are the same on every call"
    (define model (load-model tiny-dir))
    (define ids '(1))
    (define once (greedy-ids model ids 4))
    (define twice (greedy-ids model ids 4))
    (check-equal? twice once)
    (check-equal? (length once) 4))

  (test-case "each configuration is a different process"
    (define a (call-bench-child '("--child" "pid")))
    (define b (call-bench-child '("--child" "pid")))
    (check-equal? (hash-ref a 'kind) "pid")
    (check-not-equal? (hash-ref a 'pid) (hash-ref b 'pid))
    (define report
      (build-report (hasheq 'pid 11 'median_seconds 1.0 'resident_bytes 10 'peak_bytes 20)
                    (list (hasheq 'pid 12 'tokens 16 'median_ms 1.0 'peak_bytes 21)
                          (hasheq 'pid 13 'tokens 128 'median_ms 2.0 'peak_bytes 22)
                          (hasheq 'pid 14 'tokens 512 'median_ms 3.0 'peak_bytes 23)
                          (hasheq 'pid 15 'tokens 1000 'median_ms 4.0 'peak_bytes 24))
                    (hasheq 'pid 16 'new_tokens 128 'prompt_tokens 4 'tokens_per_sec 5.0
                            'peak_bytes 25 'ids '(11 262))
                    (list (hasheq 'pid 17 'rows 768 'in 768 'out 768 'gflops 1.5 'peak_bytes 26)
                          (hasheq 'pid 18 'rows 1 'in 768 'out 50257 'gflops 0.5 'peak_bytes 27))
                    #:label "vm-reference"
                    #:note "Reference run on the cloud VM, not the M5 Pro."
                    #:not-m5-pro #t
                    #:model-dir "models/gpt2"
                    #:prompt default-prompt
                    #:runs default-runs))
    (check-true (isolated-configurations? report))
    (check-equal? (configuration-pids report) '(11 12 13 14 15 16 17 18))
    (define text (report-markdown report))
    (check-true (string-contains? text "Reference run, not the M5 Pro."))
    (check-true (string-contains? text "prefill 16"))
    (check-true (string-contains? text "prefill 1000"))
    (check-true (string-contains? text "decode 128 (incl. 4-token prefill)"))
    (check-true (string-contains? text "include loading the weights"))
    (check-false (string-contains? text "seed:"))
    (check-true (string-contains? text "768×768×768"))
    (check-true (string-contains? text "1×768×50257"))
    (check-false (isolated-configurations?
                  (hash-set report 'decode (hash-set (hash-ref report 'decode) 'pid 11)))))

  (test-case "a child that fills stderr still returns"
    (define exec (find-system-path 'exec-file))
    (define racket (if (complete-path? exec) exec (or (find-executable-path exec) exec)))
    (define-values (proc out in err)
      (subprocess #f #f #f racket "-e"
                  "(display (make-string 200000 #\\x) (current-error-port)) (flush-output (current-error-port)) (display \"ok\\n\")"))
    (close-output-port in)
    (define result-ch (make-channel))
    (define worker
      (thread
       (lambda ()
         (channel-put result-ch
                      (with-handlers ([exn:fail? (lambda (e) (list 'exn (exn-message e)))])
                        (define-values (text err-text status)
                          (drain-subprocess-ports proc out err))
                        (list text (string-length err-text) status))))))
    (define result (sync/timeout 10 result-ch))
    (unless result
      (subprocess-kill proc #t)
      (kill-thread worker))
    (check-equal? result '("ok\n" 200000 0)))

  (test-case "the CPU name is non-empty on this host"
    (define name (host-cpu-name))
    (check-true (non-empty-string? name))
    (check-false (string=? name "unknown"))))
