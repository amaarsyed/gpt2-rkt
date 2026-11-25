#lang racket/base

;; Run one traced generation and build the versioned JSON document.
;; commitSha comes from GPT2_COMMIT_SHA or `git rev-parse HEAD`.

(require racket/string
         racket/system
         "embed.rkt"
         "model.rkt"
         "tokenizer.rkt"
         "trace.rkt")

(provide trace-generation
         git-commit-sha
         gpt2-small-meta)

(define (find-repo-root)
  (let loop ([dir (simplify-path (current-directory))])
    (if (or (directory-exists? (build-path dir ".git"))
            (file-exists? (build-path dir ".git")))
        dir
        (let ([parent (simplify-path (build-path dir 'up))])
          (if (equal? (path->string parent) (path->string dir))
              (current-directory)
              (loop parent))))))

(define (git-commit-sha)
  (define env (getenv "GPT2_COMMIT_SHA"))
  (if (and (string? env) (regexp-match? #px"^[0-9a-f]{40}$" env))
      env
      (let ([out (open-output-string)]
            [err (open-output-string)])
        (define ok
          (parameterize ([current-output-port out]
                         [current-error-port err]
                         [current-directory (find-repo-root)])
            (system* (find-executable-path "git") "rev-parse" "HEAD")))
        (define sha (string-trim (get-output-string out)))
        (unless (and ok (regexp-match? #px"^[0-9a-f]{40}$" sha))
          (error 'git-commit-sha "git rev-parse HEAD failed: ~a"
                 (string-trim (get-output-string err))))
        sha)))

(define (gpt2-small-meta model)
  (hasheq 'repo "openai-community/gpt2"
          'revision "607a30d783dfa663caf39e06633721c8d4cfcd7e"
          'nLayer (gpt2-model-n-layer model)
          'nHead (gpt2-model-n-head model)
          'nEmbd (gpt2-model-n-embd model)
          'vocabSize (gpt2-model-vocab-size model)
          'nCtx (min gpt2-context-length (gpt2-model-n-positions model))))

(define (token-records tok ids)
  (for/list ([id (in-list ids)])
    (hasheq 'id id 'text (decode tok (list id)))))

(define (trace-generation model prompt max-new
                          #:decoding [decoding "greedy"]
                          #:temperature [temperature #f]
                          #:top-k [top-k #f]
                          #:seed [seed #f]
                          #:commit-sha [commit-sha #f])
  (define obs (make-execution-trace))
  (define tok (gpt2-model-tokenizer model))
  (define ids
    (parameterize ([current-trace-observer obs])
      (encode tok prompt)))
  (define result
    (parameterize ([current-trace-observer obs])
      (model-generate model ids max-new
                      #:temperature temperature
                      #:top-k top-k
                      #:seed seed)))
  (define generated (generation-token-ids result))
  (define all (append ids generated))
  (define sha (or commit-sha (git-commit-sha)))
  (values
   (execution-trace->jsexpr
    obs
    #:commit-sha sha
    #:model (gpt2-small-meta model)
    #:run (hasheq 'prompt prompt
                  'maxNewTokens max-new
                  'decoding decoding
                  'promptTokenIds ids
                  'generatedTokenIds generated
                  'allTokenIds all
                  'tokens (token-records tok all)
                  'outputText (generation-text result)))
   result
   ids))
