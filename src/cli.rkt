#lang racket/base

;; Argument parsing for `racket main.rkt`.

(require racket/string
         "errors.rkt"
         "generate.rkt")

(provide top-help-text
         generate-help-text
         parse-generate-args
         run-cli)

(struct opt (name placeholder kind required? summary)
  #:transparent)

;; One row per flag. Help text and parsing both read this table.
(define option-table
  (list
   (opt 'model-dir "DIR" 'string #t
        "Directory of model assets")
   (opt 'prompt "TEXT" 'string #t
        "Prompt text")
   (opt 'max-new-tokens "N" 'number #t
        "Number of tokens to generate (integer >= 1)")
   (opt 'temperature "T" 'number #f
        "Sampling temperature (finite number > 0). Omit for greedy decoding")
   (opt 'top-k "K" 'number #f
        "Keep the K most likely tokens (integer >= 1). Requires temperature > 0")
   (opt 'seed "S" 'number #f
        "RNG seed (integer >= 0). Requires temperature > 0")))

(define opts-by-name
  (for/hash ([o (in-list option-table)])
    (values (symbol->string (opt-name o)) o)))

(define (format-opt-block opts)
  (string-join
   (for/list ([o (in-list opts)])
     (format "  --~a ~a\n      ~a" (opt-name o) (opt-placeholder o) (opt-summary o)))
   "\n"))

(define (required-usage)
  (string-join
   (for/list ([o (in-list option-table)]
              #:when (opt-required? o))
     (format "--~a ~a" (opt-name o) (opt-placeholder o)))
   " "))

(define top-help-text
  (string-append
   "gpt2-rkt: pure-Racket GPT-2 inference\n"
   "\n"
   "Usage: racket main.rkt <command> [options]\n"
   "\n"
   "Commands:\n"
   "  generate    Generate tokens from a prompt\n"
   "\n"
   "Run `racket main.rkt generate --help` for generate options.\n"))

(define generate-help-text
  (string-append
   (format "Usage: racket main.rkt generate ~a [options]\n" (required-usage))
   "\n"
   "Generate tokens from a prompt.\n"
   "Decoding is greedy unless --temperature is greater than 0.\n"
   "A flag value is the following argument, including text that starts with '-'.\n"
   "\n"
   "Required:\n"
   (format-opt-block (filter opt-required? option-table))
   "\n"
   "\n"
   "Sampling (optional; greedy by default):\n"
   (format-opt-block (filter (lambda (o) (not (opt-required? o))) option-table))
   "\n"
   "\n"
   "  --help, -h\n"
   "      Show this help\n"))

(define (help-flag? arg)
  (or (equal? arg "--help")
      (equal? arg "-h")))

(define (help-command? arg)
  (or (help-flag? arg)
      (equal? arg "help")))

;; Kinds are only 'string or 'number. Numeric ranges live on generate-config.
(define (parse-kind kind raw flag)
  (case kind
    [(string) raw]
    [(number)
     (define n (string->number raw))
     (unless (real? n)
       (raise-gpt2 (format "invalid --~a: ~a" flag raw)))
     n]
    [else (error 'parse-kind "unknown kind: ~a" kind)]))

;; Returns 'help, or a generate-config. Raises exn:fail:gpt2 on bad arguments.
;; `-h` and `--help` count only when they appear as flags, not as values.
(define (parse-generate-args args)
  (parse-options args))

(define (parse-options args)
  (define seen (make-hasheq))
  (let loop ([args args])
    (cond
      [(null? args) (build-config seen)]
      [(help-flag? (car args)) 'help]
      [else
       (define-values (name raw rest) (take-option args))
       (define spec (hash-ref opts-by-name name #f))
       (unless spec
         (raise-gpt2 (format "unknown flag: --~a" name)))
       (when (hash-has-key? seen (opt-name spec))
         (raise-gpt2 (format "duplicate flag: --~a" name)))
       (hash-set! seen (opt-name spec) (parse-kind (opt-kind spec) raw name))
       (loop rest)])))

(define (take-option args)
  (define arg (car args))
  (cond
    [(regexp-match #px"^--([A-Za-z0-9-]+)=(.*)$" arg)
     => (lambda (m)
          (values (cadr m) (caddr m) (cdr args)))]
    [(regexp-match #px"^--([A-Za-z0-9-]+)$" arg)
     => (lambda (m)
          (define name (cadr m))
          (when (null? (cdr args))
            (raise-gpt2 (format "missing value for --~a" name)))
          (values name (cadr args) (cddr args)))]
    [else
     (raise-gpt2 (format "unexpected argument: ~a" arg))]))

(define (build-config seen)
  (define missing
    (for/list ([o (in-list option-table)]
               #:when (and (opt-required? o)
                           (not (hash-has-key? seen (opt-name o)))))
      (format "--~a" (opt-name o))))
  (unless (null? missing)
    (raise-gpt2
     (format "missing required arguments: ~a" (string-join missing ", "))))
  (define (val name)
    (hash-ref seen name #f))
  (generate-config (val 'model-dir)
                   (val 'prompt)
                   (val 'max-new-tokens)
                   (val 'temperature)
                   (val 'top-k)
                   (val 'seed)))

(define (argv->strings argv)
  (define items
    (cond
      [(vector? argv) (vector->list argv)]
      [(list? argv) argv]
      [else (raise-argument-error 'run-cli "(or/c vector? list?)" argv)]))
  (for ([item (in-list items)])
    (unless (string? item)
      (raise-argument-error 'run-cli "string?" item)))
  items)

(define (run-cli argv
                 #:output-port [out (current-output-port)]
                 #:error-port [err (current-error-port)])
  (with-handlers ([exn:fail:gpt2?
                   (lambda (e)
                     (displayln (exn-message e) err)
                     1)])
    (dispatch (argv->strings argv) out)))

(define (dispatch args out)
  (cond
    [(null? args)
     (raise-gpt2 (string-append "missing command\n\n" top-help-text))]
    [(help-command? (car args))
     (display top-help-text out)
     0]
    [(equal? (car args) "generate")
     (define parsed (parse-generate-args (cdr args)))
     (cond
       [(eq? parsed 'help)
        (display generate-help-text out)
        0]
       [else
        (generate parsed #:output-port out)
        0])]
    [else
     (raise-gpt2
      (format "unknown command: ~a\n\n~a" (car args) top-help-text))]))
