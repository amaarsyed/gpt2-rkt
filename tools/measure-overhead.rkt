#lang racket/base

;; Time greedy generation of a prompt. The timed work is the sampling
;; forwards: one prompt forward and (max-new - 1) single-token forwards.
;; JSON export is not included.
;;
;;   racket tools/measure-overhead.rkt --mode off
;;   racket tools/measure-overhead.rkt --mode on
;;   racket tools/measure-overhead.rkt --mode on-generate
;;
;; --mode on attaches the trace observer to those same forwards.
;; --mode on-generate calls model-generate, so the total also includes
;; the top-15 softmax and the trace-only forward of the last token.
;; Prints every run, then the median.

(require racket/cmdline
         racket/flonum
         racket/list
         racket/string
         "../src/model.rkt"
         "../src/tensor.rkt"
         "../src/tokenizer.rkt"
         "../src/trace.rkt")

(define mode "off")
(define model-dir "models/gpt2")
(define prompt "Once upon a time")
(define max-new 16)
(define runs 5)
(define warmup 1)

(command-line
 #:program "measure-overhead"
 #:once-each
 [("--mode") m "off, on, or on-generate" (set! mode m)]
 [("--model-dir") dir "Model directory" (set! model-dir dir)]
 [("--prompt") text "Prompt" (set! prompt text)]
 [("--max-new-tokens") n "Greedy tokens" (set! max-new (string->number n))]
 [("--runs") n "Timed runs" (set! runs (string->number n))]
 [("--warmup") n "Untimed runs" (set! warmup (string->number n))])

(unless (member mode '("off" "on" "on-generate"))
  (error 'measure-overhead "mode must be off, on, or on-generate"))

(define (now) (current-inexact-monotonic-milliseconds))

(define (greedy-id logits)
  (define data (tensor-data logits))
  (define n (flvector-length data))
  (let loop ([i 1] [best 0] [bestv (flvector-ref data 0)])
    (if (= i n)
        best
        (let ([v (flvector-ref data i)])
          (if (fl> v bestv)
              (loop (add1 i) i v)
              (loop (add1 i) best bestv))))))

(define (median xs)
  (define ordered (sort xs <))
  (list-ref ordered (quotient (sub1 (length ordered)) 2)))

(define (time-forwards model ids)
  (define obs (and (equal? mode "on") (make-execution-trace)))
  (define cache (empty-kv-cache model #:capacity (+ (length ids) max-new)))
  (define-values (prefill decode total)
    (parameterize ([current-trace-observer obs])
      (define t0 (now))
      (define-values (logits cache1)
        (model-forward model ids cache #:last-only #t))
      (define id (greedy-id logits))
      (define t1 (now))
      (let loop ([left (sub1 max-new)] [prev id] [cache cache1])
        (unless (zero? left)
          (define-values (logits cache2)
            (model-forward model (list prev) cache #:last-only #t))
          (loop (sub1 left) (greedy-id logits) cache2)))
      (define t2 (now))
      (values (- t1 t0) (- t2 t1) (- t2 t0))))
  (values prefill decode total))

(define (time-generate model ids)
  (define obs (make-execution-trace))
  (parameterize ([current-trace-observer obs])
    (define t0 (now))
    (define result (model-generate model ids max-new))
    (define t1 (now))
    (values (execution-trace-prefill-ms obs)
            (execution-trace-decode-ms obs)
            (execution-trace-total-ms obs)
            (- t1 t0)
            (generation-token-ids result)
            (generation-text result))))

(define model (load-model model-dir))
(define ids (encode (gpt2-model-tokenizer model) prompt))
(fprintf (current-error-port) "mode ~a prompt-ids ~a\n" mode ids)

(define samples
  (cond
    [(equal? mode "on-generate")
     (for/list ([i (in-range (+ warmup runs))])
       (define-values (prefill decode total wall token-ids text)
         (time-generate model ids))
       (if (< i warmup)
           (begin
             (fprintf (current-error-port) "warmup wall ~a text ~a ids ~a\n"
                      wall text token-ids)
             #f)
           (begin
             (printf "~a ~a ~a ~a\n" prefill decode total wall)
             (flush-output)
             (list prefill decode total wall))))]
    [else
     (for/list ([i (in-range (+ warmup runs))])
       (define-values (prefill decode total) (time-forwards model ids))
       (if (< i warmup)
           (begin
             (fprintf (current-error-port) "warmup ~a ~a ~a\n" prefill decode total)
             #f)
           (begin
             (printf "~a ~a ~a\n" prefill decode total)
             (flush-output)
             (list prefill decode total))))]))

(define kept (filter values samples))
(define (col k) (map (lambda (row) (list-ref row k)) kept))
(fprintf (current-error-port) "median")
(for ([k (in-range (length (car kept)))])
  (fprintf (current-error-port) " ~a" (median (col k))))
(fprintf (current-error-port) "\n")
