#lang racket/base

(require racket/string
         "errors.rkt"
         "model-assets.rkt")

(provide generate
         generate-config
         generate-config?
         generate-config-model-dir
         generate-config-prompt
         generate-config-max-new-tokens
         generate-config-temperature
         generate-config-top-k
         generate-config-seed)

;; `temperature`, `top-k`, and `seed` are #f when omitted.
;; Temperature is stored as a flonum. Omitted or 0 means greedy decoding.
;; This guard is the only range check; the CLI parser only decodes argv.
(struct generate-config (model-dir prompt max-new-tokens temperature top-k seed)
  #:transparent
  #:guard
  (lambda (model-dir prompt max-new-tokens temperature top-k seed _name)
    (unless (non-empty-string? model-dir)
      (raise-gpt2 "model-dir must be non-empty"))
    (unless (non-empty-string? prompt)
      (raise-gpt2 "prompt must be non-empty"))
    (unless (exact-positive-integer? max-new-tokens)
      (raise-gpt2 "max-new-tokens must be an integer >= 1"))
    (define temp
      (cond
        [(not temperature) #f]
        [(and (real? temperature) (rational? temperature) (>= temperature 0))
         (real->double-flonum temperature)]
        [else
         (raise-gpt2 "temperature must be a finite number >= 0")]))
    (unless (or (not top-k) (exact-positive-integer? top-k))
      (raise-gpt2 "top-k must be an integer >= 1"))
    (unless (or (not seed) (exact-nonnegative-integer? seed))
      (raise-gpt2 "seed must be an integer >= 0"))
    (define sampling? (and temp (> temp 0.0)))
    (when (and top-k (not sampling?))
      (raise-gpt2 "top-k requires temperature > 0"))
    (when (and seed (not sampling?))
      (raise-gpt2 "seed requires temperature > 0"))
    (values model-dir prompt max-new-tokens temp top-k seed)))

(define (generate cfg)
  (unless (generate-config? cfg)
    (raise-argument-error 'generate "generate-config?" cfg))
  (define problem (missing-model-assets-message (generate-config-model-dir cfg)))
  (when problem
    (raise-gpt2 problem))
  (raise-gpt2 "model loading is not implemented"))
