#lang racket/base

;; Real-model benchmark. Each configuration runs in its own process so
;; current-memory-use 'peak is that configuration's high-water mark.
;; Prefill and decode are greedy, with one warmup and the median of N
;; runs. A run whose token ids differ from the warmup is an error.

(require json
         racket/date
         racket/file
         racket/format
         racket/future
         racket/list
         racket/os
         racket/port
         racket/runtime-path
         racket/string
         "harness.rkt"
         "machine.rkt"
         "../src/model-assets.rkt"
         "../src/model.rkt"
         "../src/tokenizer.rkt")

(provide prefill-lengths
         default-prompt
         default-runs
         default-new-tokens
         ids-of-length
         greedy-ids
         median-number
         configuration-pids
         isolated-configurations?
         report-markdown
         drain-subprocess-ports
         call-bench-child
         build-report)

(define prefill-lengths '(16 128 512 1000))
(define default-prompt "Once upon a time")
(define default-runs 3)
(define default-new-tokens 128)

(define-runtime-path self-path "run.rkt")
(define-runtime-path default-out-dir "results")

(define flag-takes-value
  (hash "--child" #t
        "--model-dir" #t
        "--runs" #t
        "--prompt" #t
        "--label" #t
        "--note" #t
        "--out-dir" #t
        "--new-tokens" #t
        "--tokens" #t
        "--rows" #t
        "--in" #t
        "--out" #t
        "--not-m5-pro" #f))

(define (fail fmt . args)
  (eprintf (string-append "bench: " (apply format fmt args) "\n"))
  (eprintf "usage: racket bench/run.rkt --model-dir models/gpt2 [--runs N]\n")
  (exit 1))

(define (check-args args)
  (let loop ([args args])
    (unless (null? args)
      (define name (car args))
      (unless (hash-has-key? flag-takes-value name)
        (fail "unexpected argument: ~a" name))
      (define takes (hash-ref flag-takes-value name))
      (when (and takes (null? (cdr args)))
        (fail "~a needs a value" name))
      (loop (if takes (cddr args) (cdr args))))))

;; The last copy of a flag wins, so a wrapper can set a default and the
;; caller can override it.
(define (flag args name default)
  (let loop ([args args] [found default])
    (cond
      [(or (null? args) (null? (cdr args))) found]
      [(string=? (car args) name) (loop (cddr args) (cadr args))]
      [else (loop (cdr args) found)])))

(define (int-flag args name default)
  (define text (flag args name #f))
  (cond
    [(not text) default]
    [else
     (define n (string->number text))
     (unless (exact-integer? n)
       (fail "~a must be an integer" name))
     n]))

(define (positive-int-flag args name default)
  (define n (int-flag args name default))
  (unless (exact-positive-integer? n)
    (fail "~a must be an integer >= 1" name))
  n)

(define (median-number nums)
  (list-ref (sort nums <) (quotient (length nums) 2)))

(define (ids-of-length base n)
  (define len (length base))
  (when (zero? len)
    (error 'bench "prompt encoded to no tokens"))
  (define vec (list->vector base))
  (for/list ([i (in-range n)])
    (vector-ref vec (remainder i len))))

(define (greedy-ids model prompt-ids n)
  (generation-token-ids
   (model-generate model prompt-ids n #:stop-at-eos? #f)))

(define (running-racket)
  (define exec (find-system-path 'exec-file))
  (if (complete-path? exec)
      exec
      (or (find-executable-path exec) exec)))

;; Read both pipes together. Draining stdout to EOF first deadlocks when
;; the child fills the stderr pipe before it closes stdout.
(define (drain-subprocess-ports proc out err)
  (define err-text (box ""))
  (define err-thread
    (thread (lambda () (set-box! err-text (port->string err)))))
  (define text (port->string out))
  (thread-wait err-thread)
  (close-input-port out)
  (close-input-port err)
  (subprocess-wait proc)
  (values text (unbox err-text) (subprocess-status proc)))

(define (call-bench-child args)
  (define-values (proc out in err)
    (apply subprocess #f #f #f
           (running-racket)
           (path->string self-path)
           args))
  (close-output-port in)
  (define-values (text err-text status) (drain-subprocess-ports proc out err))
  (unless (zero? status)
    (eprintf "~a" err-text)
    (error 'bench "child failed: ~a" args))
  (define js (read-json (open-input-string text)))
  (unless (hash? js)
    (error 'bench "child did not print a JSON object"))
  js)

(define (emit js)
  (write-json js)
  (newline)
  (flush-output))

(define (require-model-dir dir)
  (define problem (missing-model-assets-message dir))
  (when problem
    (eprintf "~a\n" problem)
    (exit 1))
  dir)

;; load-model does not collect. The resident figure is current-memory-use
;; after a collection, so the weight-file buffer is not part of it.
(define (load-for-bench dir)
  (define model (load-model dir))
  (define seconds (load-report-seconds (current-load-report)))
  (collect-garbage)
  (values model seconds (current-memory-use)))

(define (timed-same-ids thunk runs)
  (define warm (thunk))
  (collect-garbage)
  (define samples
    (for/list ([_ (in-range runs)])
      (define t0 (current-inexact-monotonic-milliseconds))
      (define got (thunk))
      (define ms (- (current-inexact-monotonic-milliseconds) t0))
      (unless (equal? got warm)
        (eprintf "bench: token ids changed between runs\n")
        (exit 1))
      ms))
  (values (median-number samples) warm))

(define (child-load args)
  (define dir (require-model-dir (flag args "--model-dir" "models/gpt2")))
  (define runs (positive-int-flag args "--runs" default-runs))
  (define-values (warm-model warm-seconds warm-resident) (load-for-bench dir))
  (void warm-resident)
  (set! warm-model #f)
  (define samples
    (for/list ([_ (in-range runs)])
      (define-values (model seconds resident) (load-for-bench dir))
      (set! model #f)
      (cons seconds resident)))
  (emit (hasheq 'kind "load"
                'pid (getpid)
                'warmup_seconds warm-seconds
                'seconds (map car samples)
                'median_seconds (median-number (map car samples))
                'resident_bytes (median-number (map cdr samples))
                'peak_bytes (current-memory-use 'peak))))

(define (prompt-ids dir prompt)
  (define model (load-model dir))
  (define ids (encode (gpt2-model-tokenizer model) prompt))
  (values model ids))

(define (child-prefill args)
  (define dir (require-model-dir (flag args "--model-dir" "models/gpt2")))
  (define runs (positive-int-flag args "--runs" default-runs))
  (define tokens (positive-int-flag args "--tokens" 16))
  (define prompt (flag args "--prompt" default-prompt))
  (define-values (model base) (prompt-ids dir prompt))
  (define ids (ids-of-length base tokens))
  (define-values (median-ms next-id)
    (timed-same-ids (lambda () (car (greedy-ids model ids 1))) runs))
  (emit (hasheq 'kind "prefill"
                'pid (getpid)
                'tokens tokens
                'median_ms median-ms
                'next_id next-id
                'peak_bytes (current-memory-use 'peak))))

(define (child-decode args)
  (define dir (require-model-dir (flag args "--model-dir" "models/gpt2")))
  (define runs (positive-int-flag args "--runs" default-runs))
  (define new-tokens (positive-int-flag args "--new-tokens" default-new-tokens))
  (define prompt (flag args "--prompt" default-prompt))
  (define-values (model base) (prompt-ids dir prompt))
  (define-values (median-ms ids)
    (timed-same-ids (lambda () (greedy-ids model base new-tokens)) runs))
  (define seconds (/ median-ms 1000.0))
  (emit (hasheq 'kind "decode"
                'pid (getpid)
                'prompt_tokens (length base)
                'new_tokens (length ids)
                'median_ms median-ms
                'median_seconds seconds
                'tokens_per_sec (if (> seconds 0.0) (/ (length ids) seconds) "inf")
                'ids ids
                'peak_bytes (current-memory-use 'peak))))

(define (child-matmul args)
  (define rows (positive-int-flag args "--rows" 768))
  (define in (positive-int-flag args "--in" 768))
  (define out (positive-int-flag args "--out" 768))
  (define runs (positive-int-flag args "--runs" default-runs))
  (define sample (run-matmul-benchmark #:rows rows #:in in #:out out #:runs runs))
  (define gflops (matmul-sample-gflops sample))
  (emit (hasheq 'kind "matmul"
                'pid (getpid)
                'rows rows
                'in in
                'out out
                'median_ms (matmul-sample-ms sample)
                'gflops (if (and (real? gflops) (rational? gflops)) gflops "inf")
                'peak_bytes (current-memory-use 'peak))))

(define (child-pid)
  (emit (hasheq 'kind "pid" 'pid (getpid))))

(define (configuration-pids report)
  (append (list (hash-ref (hash-ref report 'load) 'pid))
          (map (lambda (row) (hash-ref row 'pid)) (hash-ref report 'prefill))
          (list (hash-ref (hash-ref report 'decode) 'pid))
          (map (lambda (row) (hash-ref row 'pid)) (hash-ref report 'matmul))))

(define (isolated-configurations? report)
  (define pids (configuration-pids report))
  (and (andmap exact-positive-integer? pids)
       (= (length pids) (length (remove-duplicates pids)))))

(define (build-report load-js prefills decode-js matmuls
                      #:label label
                      #:note note
                      #:not-m5-pro not-m5?
                      #:model-dir model-dir
                      #:prompt prompt
                      #:runs runs)
  (define day (current-date))
  (hasheq 'label label
          'not_m5_pro (if not-m5? #t #f)
          'note note
          'date (format "~a-~a-~a"
                        (date-year day)
                        (~r (date-month day) #:min-width 2 #:pad-string "0")
                        (~r (date-day day) #:min-width 2 #:pad-string "0"))
          'git_sha (host-git-sha)
          'racket_version (version)
          'vm (symbol->string (system-type 'vm))
          'os (symbol->string (system-type 'os))
          'arch (symbol->string (system-type 'arch))
          'processor_count (processor-count)
          'cpu (host-cpu-name)
          'physical_cpu (or (host-physical-cpu) 'null)
          'ncpu (or (host-ncpu) 'null)
          'model_dir model-dir
          'prompt prompt
          'decoding "greedy"
          'runs runs
          'warmup 1
          'load load-js
          'prefill prefills
          'decode decode-js
          'matmul matmuls))

(define (slug text)
  (string-trim (regexp-replace* #px"[^A-Za-z0-9]+" text "-") "-"))

(define (fmt-num n)
  (cond
    [(and (real? n) (inexact? n)) (~r n #:precision '(= 4))]
    [(and (real? n) (rational? n)) (~r (exact->inexact n) #:precision '(= 4))]
    [else (format "~a" n)]))

(define (fmt-bytes n)
  (if (exact-integer? n)
      (~r n #:group-sep ",")
      (format "~a" n)))

(define (report-markdown report)
  (define load (hash-ref report 'load))
  (define decode (hash-ref report 'decode))
  (define rows
    (append
     (list (list "load"
                 (format "~a s" (fmt-num (hash-ref load 'median_seconds)))
                 (hash-ref load 'peak_bytes)
                 (hash-ref load 'pid)))
     (for/list ([row (in-list (hash-ref report 'prefill))])
       (list (format "prefill ~a" (hash-ref row 'tokens))
             (format "~a ms" (fmt-num (hash-ref row 'median_ms)))
             (hash-ref row 'peak_bytes)
             (hash-ref row 'pid)))
     (list (list (format "decode ~a (incl. ~a-token prefill)"
                        (hash-ref decode 'new_tokens)
                        (hash-ref decode 'prompt_tokens))
                 (format "~a tok/s" (fmt-num (hash-ref decode 'tokens_per_sec)))
                 (hash-ref decode 'peak_bytes)
                 (hash-ref decode 'pid)))
     (for/list ([row (in-list (hash-ref report 'matmul))])
       (list (format "matmul ~a×~a×~a"
                     (hash-ref row 'rows) (hash-ref row 'in) (hash-ref row 'out))
             (format "~a GFLOP/s" (fmt-num (hash-ref row 'gflops)))
             (hash-ref row 'peak_bytes)
             (hash-ref row 'pid)))))
  (define lines
    (append
     (if (hash-ref report 'not_m5_pro)
         '("Reference run, not the M5 Pro." "")
         '())
     (list (format "label: ~a" (hash-ref report 'label))
           (format "date: ~a" (hash-ref report 'date))
           (format "cpu: ~a" (hash-ref report 'cpu))
           (format "physical-cpu: ~a" (json-text (hash-ref report 'physical_cpu)))
           (format "ncpu: ~a" (json-text (hash-ref report 'ncpu)))
           (format "processors: ~a" (hash-ref report 'processor_count))
           (format "racket: ~a (~a)" (hash-ref report 'racket_version) (hash-ref report 'vm))
           (format "git: ~a" (hash-ref report 'git_sha))
           (format "prompt: ~a" (hash-ref report 'prompt))
           (format "decoding: ~a, warmup ~a, median of ~a"
                   (hash-ref report 'decoding)
                   (hash-ref report 'warmup)
                   (hash-ref report 'runs))
           (format "note: ~a" (hash-ref report 'note))
           ""
           "Each row is its own process. Peak is current-memory-use 'peak in that process. Load, prefill, and decode peaks include loading the weights."
           ""
           "| configuration | median | peak bytes | pid |"
           "| --- | --- | ---: | ---: |")
     (for/list ([row (in-list rows)])
       (format "| ~a | ~a | ~a | ~a |"
               (car row) (cadr row) (fmt-bytes (caddr row)) (cadddr row)))
     (list ""
           (format "load resident after collection: ~a bytes"
                   (fmt-bytes (hash-ref load 'resident_bytes)))
           (format "decode produced ~a ids, first ~a."
                   (hash-ref decode 'new_tokens)
                   (if (null? (hash-ref decode 'ids))
                       "none"
                       (car (hash-ref decode 'ids)))))))
  (string-join lines "\n" #:after-last "\n"))

(define (json-text value)
  (cond
    [(eq? value 'null) "unavailable"]
    [(not value) "unavailable"]
    [else (format "~a" value)]))

(define (write-pretty js out indent)
  (define pad (make-string indent #\space))
  (define (scalar? value)
    (or (string? value)
        (and (number? value) (rational? value))
        (boolean? value)
        (eq? value 'null)
        (and (list? value)
             (andmap (lambda (item)
                       (or (and (number? item) (rational? item))
                           (string? item)
                           (eq? item 'null)))
                     value))))
  (cond
    [(eq? js 'null) (display "null" out)]
    [(boolean? js) (display (if js "true" "false") out)]
    [(scalar? js) (write-json js out)]
    [(hash? js)
     (define keys (sort (hash-keys js) symbol<?))
     (display "{\n" out)
     (for ([key (in-list keys)] [i (in-naturals)])
       (display (make-string (+ indent 2) #\space) out)
       (write (symbol->string key) out)
       (display ": " out)
       (write-pretty (hash-ref js key) out (+ indent 2))
       (unless (= i (sub1 (length keys))) (display "," out))
       (newline out))
     (display pad out)
     (display "}" out)]
    [(list? js)
     (display "[\n" out)
     (for ([item (in-list js)] [i (in-naturals)])
       (display (make-string (+ indent 2) #\space) out)
       (write-pretty item out (+ indent 2))
       (unless (= i (sub1 (length js))) (display "," out))
       (newline out))
     (display pad out)
     (display "]" out)]
    [else (write-json js out)]))

(define (write-report report out-dir)
  (make-directory* out-dir)
  (define stem (format "~a-~a" (hash-ref report 'label) (hash-ref report 'date)))
  (define json-path (build-path out-dir (string-append stem ".json")))
  (define md-path (build-path out-dir (string-append stem ".md")))
  (call-with-output-file json-path
    (lambda (out)
      (write-pretty report out 0)
      (newline out))
    #:exists 'replace)
  (call-with-output-file md-path
    (lambda (out) (display (report-markdown report) out))
    #:exists 'replace)
  (values json-path md-path))

(define (run-parent args)
  (define model-dir (flag args "--model-dir" "models/gpt2"))
  (require-model-dir model-dir)
  (define runs (positive-int-flag args "--runs" default-runs))
  (define prompt (flag args "--prompt" default-prompt))
  (define new-tokens (positive-int-flag args "--new-tokens" default-new-tokens))
  (define label (flag args "--label" (slug (host-cpu-name))))
  (define note (flag args "--note" ""))
  (define not-m5? (member "--not-m5-pro" args))
  (define out-dir (flag args "--out-dir" (path->string default-out-dir)))
  (define common
    (list "--model-dir" model-dir
          "--runs" (number->string runs)
          "--prompt" prompt))
  (define (step name arg-list)
    (eprintf "bench: ~a\n" name)
    (flush-output (current-error-port))
    (call-bench-child arg-list))
  (define load-js (step "load" (append '("--child" "load") common)))
  (define prefills
    (for/list ([n (in-list prefill-lengths)])
      (step (format "prefill ~a" n)
            (append (list "--child" "prefill" "--tokens" (number->string n))
                    common))))
  (define decode-js
    (step (format "decode ~a" new-tokens)
          (append (list "--child" "decode" "--new-tokens" (number->string new-tokens))
                  common)))
  (define matmuls
    (for/list ([shape (in-list '((768 768 768) (1 768 50257)))])
      (step (format "matmul ~a" shape)
            (list "--child" "matmul"
                  "--rows" (number->string (car shape))
                  "--in" (number->string (cadr shape))
                  "--out" (number->string (caddr shape))
                  "--runs" (number->string runs)))))
  (define report
    (build-report load-js prefills decode-js matmuls
                  #:label label
                  #:note note
                  #:not-m5-pro not-m5?
                  #:model-dir model-dir
                  #:prompt prompt
                  #:runs runs))
  (unless (isolated-configurations? report)
    (fail "configurations shared a process: ~a" (configuration-pids report)))
  (define-values (json-path md-path) (write-report report out-dir))
  (display (report-markdown report))
  (eprintf "wrote ~a\nwrote ~a\n" json-path md-path))

(module+ main
  (define args (vector->list (current-command-line-arguments)))
  (check-args args)
  (define child (flag args "--child" #f))
  (cond
    [(not child) (run-parent args)]
    [(string=? child "pid") (child-pid)]
    [(string=? child "load") (child-load args)]
    [(string=? child "prefill") (child-prefill args)]
    [(string=? child "decode") (child-decode args)]
    [(string=? child "matmul") (child-matmul args)]
    [else (fail "unknown child: ~a" child)]))
