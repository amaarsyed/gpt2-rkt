#lang racket/base

(module+ test
  (require json
           racket/flonum
           racket/runtime-path
           racket/string
           rackunit
           "../main.rkt"
           "../src/model.rkt"
           "fixture-util.rkt")

  (define-runtime-path tiny-dir "tiny-gpt2")

  (define (last xs)
    (if (null? (cdr xs)) (car xs) (last (cdr xs))))

  (define (argmax-id nums)
    (let loop ([i 1] [best 0] [bestv (car nums)])
      (if (= i (length nums))
          best
          (let ([v (list-ref nums i)])
            (if (> v bestv)
                (loop (add1 i) i v)
                (loop (add1 i) best bestv))))))

  (define (last-row-nums logits)
    (define rows (car (tensor-dims logits)))
    (define cols (cadr (tensor-dims logits)))
    (define data (tensor-data logits))
    (define base (* (sub1 rows) cols))
    (for/list ([j (in-range cols)])
      (flvector-ref data (+ base j))))

  (define (flvector->list data)
    (for/list ([i (in-range (flvector-length data))])
      (flvector-ref data i)))

  (test-case "tiny eager logits and the greedy continuation match"
    (define ref (call-with-input-file (build-path tiny-dir "reference.json") read-json))
    (define model (load-model tiny-dir))
    (define prompt-ids (hash-ref ref 'prompt_ids))
    (check-equal? (encode (gpt2-model-tokenizer model) (hash-ref ref 'prompt))
                  prompt-ids)
    (define steps (hash-ref ref 'steps))
    (define cache (empty-kv-cache model))
    (define previous #f)
    (for ([step (in-list steps)])
      (define context (hash-ref step 'context))
      (define expected (flat-flonums (hash-ref step 'logits)))
      (define-values (logits cache*)
        (if previous
            (model-forward model (list (last context)) cache)
            (model-forward model context cache)))
      (set! cache cache*)
      (define got (last-row-nums logits))
      (define full (flvector->list (next-token-logits model context)))
      (check-equal? got full)
      (check-true (within-tolerance? got expected))
      (check-equal? (hash-ref step 'next) (argmax-id got))
      (set! previous context))
    (check-equal? (generation-token-ids
                   (model-generate model prompt-ids (hash-ref ref 'max_new_tokens)))
                  (hash-ref ref 'generated_ids))
    (define out (open-output-string))
    (define err (open-output-string))
    (define code
      (run-cli (list "generate" "--model-dir" (path->string tiny-dir)
                     "--prompt" (hash-ref ref 'prompt)
                     "--max-new-tokens" (number->string (hash-ref ref 'max_new_tokens)))
               #:output-port out
               #:error-port err))
    (check-equal? code 0)
    (check-equal? (get-output-string err) "")
    (check-equal? (get-output-string out)
                  (string-append
                   (generation-text
                    (model-generate model prompt-ids (hash-ref ref 'max_new_tokens)))
                   "\n")))
  )