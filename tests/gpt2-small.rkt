#lang racket/base

;; Compare GPT-2 Small with the pinned eager fixture. The acceptance
;; bar is |a-e| <= 1e-3 + 1e-4*|e|. Missing assets skip the module so a
;; checkout without the download still passes.

(module+ test
  (require json
           racket/file
           racket/flonum
           racket/list
           racket/port
           racket/runtime-path
           racket/string
           rackunit
           "../main.rkt"
           "../src/embed.rkt"
           "../src/errors.rkt"
           "../src/model-assets.rkt"
           "../src/model.rkt"
           "../src/tensor.rkt"
           "fixture-util.rkt")

  (define-runtime-path model-dir "../models/gpt2")
  (define-runtime-path digest-path "../tools/gpt2-assets.sha256")
  (define-runtime-path cases-path "gpt2-small-cases.json")

  (define (asset-ready? name)
    (file-exists? (build-path model-dir name)))

  (define (entry-name key)
    (if (symbol? key) (symbol->string key) key))

  (define (read-header path)
    (call-with-input-file path
      (lambda (in)
        (define len (integer-bytes->integer (read-bytes 8 in) #f #f))
        (values len (read-json (open-input-bytes (read-bytes len in)))))
      #:mode 'binary))

  (define (digest-table path)
    (call-with-input-file path
      (lambda (in)
        (for/fold ([table (hash)]) ([line (in-lines in)])
          (define parts (string-split line))
          (if (and (>= (length parts) 2) (not (string-prefix? (car parts) "#")))
              (hash-set table (cadr parts) (car parts))
              table)))))

  (define (stage-error got stats)
    (define data (tensor-data got))
    (define n (flvector-length data))
    (define-values (mn mx sum)
      (for/fold ([mn +inf.0] [mx -inf.0] [sum 0.0])
                ([i (in-range n)])
        (define v (flvector-ref data i))
        (values (min mn v) (max mx v) (+ sum v))))
    (define pairs
      (cons (list mn (hash-ref stats 'min))
            (cons (list mx (hash-ref stats 'max))
                  (cons (list (/ sum n) (hash-ref stats 'mean))
                        (for/list ([sample (in-list (hash-ref stats 'samples))])
                          (list (flvector-ref data (hash-ref sample 'i))
                                (hash-ref sample 'v)))))))
    (check-equal? n (hash-ref stats 'count))
    (for/fold ([worst 0.0]) ([pair (in-list pairs)])
      (define got-v (car pair))
      (define exp-v (real->double-flonum (cadr pair)))
      (check-true (within-tolerance? (list got-v) (list exp-v)))
      (max worst (abs (- got-v exp-v)))))

  (cond
    [(not (andmap asset-ready? required-model-asset-names))
     (printf "skipping GPT-2 Small: ~a is missing files\n" model-dir)]
    [else
     (define model (load-model model-dir))
     (define loaded (current-load-report))
     (printf "load-seconds ~a load-bytes ~a\n"
             (load-report-seconds loaded)
             (load-report-bytes loaded))
     (define cases (call-with-input-file cases-path read-json))
     (define-values (header-len header) (read-header (build-path model-dir "model.safetensors")))

     (test-case "the pinned files match the published digests"
       (define digests (digest-table digest-path))
       (for ([name (in-list required-model-asset-names)])
         (check-not-exn
          (lambda () (check-file-sha256 (build-path model-dir name) (hash-ref digests name))))))

     (test-case "the official header is unaligned and the mask buffers are ignored"
       (check-equal? (remainder header-len 8) 3)
       (define names
         (for/list ([(key info) (in-hash header)])
           (define name (entry-name key))
           (unless (equal? name "__metadata__")
             (check-equal? (hash-ref info 'dtype) "F32"))
           name))
       (check-equal? (length (filter (lambda (name) (regexp-match? #rx"^h\\.[0-9]+\\.attn\\.bias$" name))
                                     names))
                     12)
       (check-equal? (hash-ref (hash-ref header 'wte.weight) 'shape) '(50257 768))
       (check-equal? (hash-ref (hash-ref header 'h.0.attn.c_attn.weight) 'shape) '(768 2304))
       (check-false (member "lm_head.weight" names))
       (check-equal? (gpt2-model-n-positions model) gpt2-context-length)
       (check-equal? (gpt2-model-n-embd model) 768)
       (check-equal? (gpt2-model-vocab-size model) 50257)
       (check-exn #rx"1024"
                  (lambda () (next-token-logits model (make-list 1025 0)))))

     (test-case "eager hidden states, logits, and greedy tokens match"
       (for ([prompt (in-list (hash-ref cases 'prompts))])
         (define text (hash-ref prompt 'text))
         (define ids (hash-ref prompt 'ids))
         (check-equal? (encode (gpt2-model-tokenizer model) text) ids)
         (define trace (model-activations model ids))
         (define stages (hash-ref prompt 'stages))
         (define blocks (activation-trace-blocks trace))
         (define worsts
           (list (list "embed" (stage-error (activation-trace-embed trace) (hash-ref stages 'embed)))
                 (list "block0" (stage-error (list-ref blocks 0) (hash-ref stages 'block0)))
                 (list "block5" (stage-error (list-ref blocks 5) (hash-ref stages 'block5)))
                 (list "block11" (stage-error (list-ref blocks 11) (hash-ref stages 'block11)))
                 (list "ln_f" (stage-error (activation-trace-ln-f trace) (hash-ref stages 'ln_f)))))
         (define logits (tensor-data (activation-trace-logits trace)))
         (define logit-worst
           (for/fold ([worst 0.0]) ([item (in-list (hash-ref prompt 'top_logits))])
             (define got (flvector-ref logits (hash-ref item 'i)))
             (define exp-v (real->double-flonum (hash-ref item 'v)))
             (check-true (within-tolerance? (list got) (list exp-v)))
             (max worst (abs (- got exp-v)))))
         (printf "~a stage max-abs ~a logit max-abs ~a\n" text worsts logit-worst)
         (define started (current-inexact-milliseconds))
         (define generated
           (generation-token-ids
            (model-generate model ids (hash-ref cases 'max_new_tokens))))
         (define seconds (/ (- (current-inexact-milliseconds) started) 1000.0))
         (printf "~a generate-seconds ~a tokens-per-second ~a\n"
                 text seconds (/ (length generated) seconds))
         (check-equal? generated (hash-ref prompt 'greedy_ids))))

     (test-case "the CLI prints the eager greedy text"
       (define prompt (findf (lambda (item) (string=? (hash-ref item 'text) "Once upon a time"))
                             (hash-ref cases 'prompts)))
       (define out (open-output-string))
       (define err (open-output-string))
       (define code
         (run-cli (list "generate"
                        "--model-dir" (path->string model-dir)
                        "--prompt" "Once upon a time"
                        "--max-new-tokens" "16")
                  #:output-port out
                  #:error-port err))
       (check-equal? code 0)
       (check-equal? (get-output-string out)
                     (string-append (hash-ref prompt 'greedy_text) "\n"))
       (check-regexp-match #px"tokens-per-second: " (get-output-string err)))]))
