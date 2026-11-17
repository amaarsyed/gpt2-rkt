#lang racket/base

;; One driver for both historical engines. Only the child imports the
;; selected snapshot. Each configuration gets a fresh process.

(require json
         racket/cmdline
         racket/file
         racket/flonum
         racket/format
         racket/future
         racket/list
         racket/os
         racket/runtime-path
         racket/string
         "machine.rkt"
         (only-in "run.rkt" drain-subprocess-ports)
         "../src/model-assets.rkt")

(define-runtime-path repo-root "..")
(define-runtime-path self "compare-phases.rkt")
(define phase-revisions
  '(("phase6" . "4e0631f4bb12ebd9e11182174de6ec808195568d")
    ("phase7" . "0f32e9b23836e07dd18c7d83c00caf84bb2c1e0b")))
(define prompt "Once upon a time")

(define (median xs)
  (list-ref (sort xs <) (quotient (length xs) 2)))

(define (repeat-ids ids n)
  (define v (list->vector ids))
  (for/list ([i (in-range n)])
    (vector-ref v (modulo i (vector-length v)))))

(define (timed thunk result->ids runs)
  (define warm (result->ids (thunk)))
  (collect-garbage)
  (define allocated '())
  (define gc-times '())
  (define samples
    (for/list ([_ (in-range runs)])
      (define bytes (current-memory-use 'cumulative))
      (define gc (current-gc-milliseconds))
      (define start (current-inexact-monotonic-milliseconds))
      (define result (thunk))
      (define ms (- (current-inexact-monotonic-milliseconds) start))
      (set! allocated (cons (- (current-memory-use 'cumulative) bytes) allocated))
      (set! gc-times (cons (- (current-gc-milliseconds) gc) gc-times))
      (unless (equal? warm (result->ids result))
        (error 'compare-phases "result changed between warmup and timed run"))
      ms))
  (values samples warm (reverse allocated) (reverse gc-times)))

(define (child source model-dir kind n runs smoke? decode-workers prefill-workers)
  (define (binding module name)
    (dynamic-require (build-path source "src" module) name))
  (define load-model (binding "model.rkt" 'load-model))
  (define vocab-size (binding "model.rkt" 'gpt2-model-vocab-size))
  (define model-tokenizer (binding "model.rkt" 'gpt2-model-tokenizer))
  (define model-generate (binding "model.rkt" 'model-generate))
  (define token-ids (binding "model.rkt" 'generation-token-ids))
  (define encode (binding "tokenizer.rkt" 'encode))
  (define make-tensor (binding "tensor.rkt" 'make-tensor))
  (define tensor-data (binding "tensor.rkt" 'tensor-data))
  (define matmul (binding "nn.rkt" 'matmul))
  (define head (binding "nn.rkt" 'matmul-transposed-b))
  (when decode-workers ((binding "nn.rkt" 'current-decode-workers) decode-workers))
  (when prefill-workers ((binding "nn.rkt" 'current-prefill-workers) prefill-workers))
  (define result
    (cond
      [(equal? kind "load")
       (define (once)
         (collect-garbage)
         (define start (current-inexact-monotonic-milliseconds))
         (define model (load-model model-dir))
         (define ms (- (current-inexact-monotonic-milliseconds) start))
         (collect-garbage)
         (define resident (current-memory-use))
         ;; Keep the model live across the collection and memory reading.
         (unless (positive? (vocab-size model)) (error 'compare-phases "empty model"))
         (list ms resident))
       (once)
       (define samples (for/list ([_ (in-range runs)]) (once)))
       (hasheq 'samples_ms (map car samples)
               'resident_samples_bytes (map cadr samples)
               'median_ms (median (map car samples))
               'resident_bytes (median (map cadr samples)))]
      [(member kind '("prefill" "decode" "sample"))
       (define model (load-model model-dir))
       (define base (encode (model-tokenizer model) (if smoke? "hi" prompt)))
       (define ids (if (equal? kind "prefill") (repeat-ids base n) base))
       (define count (if (equal? kind "prefill") 1 n))
       ;; No stop-at-eos override: phase 6 does not have that keyword.
       ;; Both revisions must produce the complete requested token count.
       (define-values (samples warm allocated gc-times)
         (timed (lambda ()
                  (if (equal? kind "sample")
                      (model-generate model ids count #:temperature 0.8 #:top-k 40 #:seed 1)
                      (model-generate model ids count))) token-ids runs))
       (unless (= (length warm) count)
         (error 'compare-phases "generation stopped early: ~a of ~a tokens" (length warm) count))
       (hasheq 'samples_ms samples 'median_ms (median samples)
               'samples_allocated_bytes allocated 'samples_gc_ms gc-times
               'prompt_tokens (length ids) 'new_tokens count 'ids warm)]
      [else
       (define rows (if (equal? kind "square") n 1))
       (define in (if smoke? 16 768))
       (define out (if (equal? kind "square") in (if smoke? 97 50257)))
       (define x (make-tensor (list rows in) (make-flvector (* rows in) 1.0)))
       (define W (make-tensor (if (equal? kind "head") (list out in) (list in out))
                              (make-flvector (* in out) 1.0)))
       (define b (make-tensor (list out) (make-flvector out 0.0)))
       (define (check result)
         (define data (tensor-data result))
         (unless (and (= (flvector-ref data 0) in)
                      (= (flvector-ref data (sub1 (flvector-length data))) in))
           (error 'compare-phases "matrix product did not match"))
         #t)
       (define-values (samples _ allocated gc-times)
         (timed (lambda () (if (equal? kind "head") (head x W) (matmul x W b))) check runs))
       (hasheq 'samples_ms samples 'median_ms (median samples)
               'samples_allocated_bytes allocated 'samples_gc_ms gc-times
               'rows rows 'inner in 'cols out)]))
  (hash-set (hash-set result 'peak_bytes (current-memory-use 'peak)) 'pid (getpid)))

(define (run-child source model-dir kind n runs smoke?)
  (define exe (find-system-path 'exec-file))
  (define racket (if (complete-path? exe) exe (find-executable-path exe)))
  (define args
    (append (list (path->string self) "--source" (path->string source)
                  "--model-dir" (path->string model-dir) "--child" kind
                  "--size" (number->string n) "--runs" (number->string runs))
            (if smoke? '("--smoke") '())))
  (define-values (proc out in err) (apply subprocess #f #f #f racket args))
  (close-output-port in)
  (define-values (text errors status) (drain-subprocess-ports proc out err))
  (unless (zero? status)
    (error 'compare-phases "~a failed: ~a" kind errors))
  (read-json (open-input-string text)))

(define (write-result path result)
  (call-with-output-file path
    (lambda (out) (write-json result out) (newline out))
    #:exists 'error))

(define (run-phase phase sha model-dir out-dir date cpu runs smoke?)
  (define source (build-path repo-root "models" "phase-bench" phase))
  (unless (file-exists? (build-path source "src" "model.rkt"))
    (error 'compare-phases "missing snapshot: ~a" source))
  (define jobs
    (if smoke?
        '(("load" 0) ("prefill" 16) ("decode" 4) ("square" 16) ("logits" 1))
        '(("load" 0) ("prefill" 16) ("prefill" 128) ("prefill" 512)
          ("prefill" 1000) ("decode" 128) ("square" 768) ("logits" 1))))
  (define results
    (for/list ([job (in-list jobs)])
      (define kind (car job))
      (define n (cadr job))
      (eprintf "~a: ~a ~a\n" phase kind n)
      (flush-output (current-error-port))
      (define row (run-child source model-dir kind n runs smoke?))
      (define result (hash-set (hash-set row 'kind kind) 'size n))
      ;; Save each completed row so an interrupted long run retains evidence.
      (write-result (build-path out-dir (format "~a-~a-~a-~a.json" phase kind n date)) result)
      (eprintf "~a: ~a ~a median ~a ms\n" phase kind n (~r (hash-ref row 'median_ms) #:precision 3))
      result))
  (define report
    (hasheq 'phase phase 'source_sha sha 'harness_sha (host-git-sha)
            'date date 'cpu cpu 'os (symbol->string (system-type 'os))
            'arch (symbol->string (system-type 'arch)) 'processors (processor-count)
            'racket (version) 'vm (symbol->string (system-type 'vm))
            'prompt (if smoke? "hi" prompt) 'runs runs 'warmup 1
            'smoke smoke? 'results results))
  (write-result (build-path out-dir (format "~a-windows-~a.json" phase date)) report)
  report)

(define (comparison before after)
  (unless (= (length (hash-ref before 'results)) (length (hash-ref after 'results)))
    (error 'compare-phases "revisions have different result counts"))
  (define lines
    (for/list ([a (in-list (hash-ref before 'results))]
               [b (in-list (hash-ref after 'results))])
      (unless (and (equal? (hash-ref a 'kind) (hash-ref b 'kind))
                   (= (hash-ref a 'size) (hash-ref b 'size))
                   (equal? (hash-ref a 'ids #f) (hash-ref b 'ids #f)))
        (error 'compare-phases "revisions produced different configurations or token IDs"))
      (define ams (hash-ref a 'median_ms))
      (define bms (hash-ref b 'median_ms))
      (format "| ~a ~a | ~a | ~a | ~a | ~a | ~a |"
              (hash-ref a 'kind) (hash-ref a 'size)
              (~r ams #:precision '(= 3)) (~r bms #:precision '(= 3))
              (~r (/ ams bms) #:precision '(= 2))
              (hash-ref a 'peak_bytes) (hash-ref b 'peak_bytes))))
  (string-join
   (append
    (list "# Phase 6 and phase 7 on the same machine" ""
          (format "Date: ~a. CPU: ~a. Racket: ~a CS. Processors: ~a."
                  (hash-ref before 'date) (hash-ref before 'cpu)
                  (hash-ref before 'racket) (hash-ref before 'processors))
          (format "Phase 6: `~a`. Phase 7: `~a`."
                  (hash-ref before 'source_sha) (hash-ref after 'source_sha))
          (format "Common driver: `~a`." (hash-ref before 'harness_sha))
          ""
          (format "One warmup, then the median of ~a calls. Each row uses a fresh process."
                  (hash-ref before 'runs))
          "Times are milliseconds. Speedup is phase 6 time / phase 7 time; above 1 is faster."
          "Peak bytes are Racket's process high-water mark, including model loading, not OS working set."
          "Load time excludes the collection used to measure resident bytes. Filesystem caches are warm."
          (if (hash-ref before 'smoke)
              "Smoke check only: tiny weights and small configurations, not a performance result."
              "Prefill includes one generated token. Decode includes the four-token prefill and 128 new tokens.")
          "Both phases use their original EOS behavior; full requested counts and identical token IDs are checked."
          ""
          "| Configuration | Phase 6 ms | Phase 7 ms | Speedup | Phase 6 peak bytes | Phase 7 peak bytes |"
          "| --- | ---: | ---: | ---: | ---: | ---: |")
    lines
    (list "" "The JSON files contain every timed sample and the generated token IDs."
          "Runs are sequential: phase 6 first, then phase 7. Thermal and background-load effects are not controlled."
          "These are Windows measurements, not the existing cloud VM or an M5 Pro benchmark."))
   "\n" #:after-last "\n"))

(module+ main
  (define source #f)
  (define kind #f)
  (define size 0)
  (define runs 3)
  (define decode-workers #f)
  (define prefill-workers #f)
  (define date #f)
  (define cpu (host-cpu-name))
  (define smoke? #f)
  (define model-dir (build-path repo-root "models" "gpt2"))
  (define out-dir #f)
  (command-line
   #:once-each
   [("--source") path "Child snapshot directory" (set! source (string->path path))]
   [("--child") name "Child configuration" (set! kind name)]
   [("--size") n "Child problem size" (set! size (string->number n))]
   [("--runs") n "Timed calls after one warmup" (set! runs (string->number n))]
   [("--decode-workers") n "Single-row matrix worker budget" (set! decode-workers (string->number n))]
   [("--prefill-workers") n "Multi-row matrix worker budget" (set! prefill-workers (string->number n))]
   [("--date") day "Report date in YYYY-MM-DD form" (set! date day)]
   [("--cpu") name "CPU model for the report" (set! cpu name)]
   [("--model-dir") path "Shared pinned model directory" (set! model-dir (path->complete-path path))]
   [("--out-dir") path "New output directory" (set! out-dir (path->complete-path path))]
   [("--smoke") "Use the tiny model and small configurations" (set! smoke? #t)])
  (unless (exact-positive-integer? runs) (error 'compare-phases "runs must be positive"))
  (cond
    [kind
     (unless (and source (member kind '("load" "prefill" "decode" "sample" "square" "logits" "head")))
       (error 'compare-phases "invalid child configuration"))
     (write-json (child source model-dir kind size runs smoke? decode-workers prefill-workers))
     (newline)]
    [else
     (unless (and date out-dir) (error 'compare-phases "--date and --out-dir are required"))
     (make-directory* out-dir)
     (unless smoke?
       (define digests (build-path repo-root "tools" "gpt2-assets.sha256"))
       (call-with-input-file digests
         (lambda (in)
           (for ([line (in-lines in)])
             (define parts (string-split line))
             (when (= (length parts) 2)
               (check-file-sha256 (build-path model-dir (cadr parts)) (car parts)))))))
     (define reports
       (for/list ([phase (in-list phase-revisions)])
         (run-phase (car phase) (cdr phase) model-dir out-dir date cpu runs smoke?)))
     (define markdown (comparison (car reports) (cadr reports)))
     (call-with-output-file (build-path out-dir (format "phase6-vs-phase7-~a.md" date))
       (lambda (out) (display markdown out)) #:exists 'error)
     (display markdown)]))

(module+ test
  (require rackunit)
  (test-case "median sorts the samples"
    (check-equal? (median '(3 1 2)) 2))
  (test-case "repeated prompt has exactly the requested tokens"
    (check-equal? (repeat-ids '(1 2 3) 5) '(1 2 3 1 2)))
  (test-case "a timed call must agree with the warmup"
    (define count 0)
    (check-exn #rx"result changed"
               (lambda () (timed (lambda () (set! count (add1 count)) count) values 1)))))
