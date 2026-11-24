#lang racket/base

;; Export one execution trace.
;;
;;   racket tools/export-trace.rkt \
;;     --model-dir models/gpt2 \
;;     --out /opt/cursor/artifacts/once-upon-a-time-16.trace.json
;;
;; GPT2_COMMIT_SHA overrides `git rev-parse HEAD`.

(require racket/cmdline
         racket/file
         racket/path
         racket/string
         "../src/model.rkt"
         "../src/trace-run.rkt"
         "../src/trace.rkt")

(define model-dir "models/gpt2")
(define out-path "/opt/cursor/artifacts/once-upon-a-time-16.trace.json")
(define prompt "Once upon a time")
(define max-new 16)

(command-line
 #:program "export-trace"
 #:once-each
 [("--model-dir") dir "Directory of model assets" (set! model-dir dir)]
 [("--out") path "Trace JSON path" (set! out-path path)]
 [("--prompt") text "Prompt text" (set! prompt text)]
 [("--max-new-tokens") n "Greedy tokens to generate"
  (set! max-new (string->number n))])

(unless (and (exact-positive-integer? max-new))
  (error 'export-trace "max-new-tokens must be an integer >= 1"))

(define model (load-model model-dir))
(define-values (doc result ids) (trace-generation model prompt max-new))
(define parent (path-only (string->path out-path)))
(when parent (make-directory* parent))
(write-execution-trace out-path doc)
(fprintf (current-error-port) "prompt-ids: ~a\n" ids)
(fprintf (current-error-port) "generated-ids: ~a\n" (generation-token-ids result))
(fprintf (current-error-port) "commit: ~a\n" (hash-ref doc 'commitSha))
(displayln (generation-text result))
