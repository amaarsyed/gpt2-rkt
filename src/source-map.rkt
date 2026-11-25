#lang racket/base

;; Compile-time pins for the execution-trace stages. Each pin reads
;; syntax-source, syntax-line, and syntax-span while the module expands,
;; so the stored file, line, and snippet are the source that was compiled.
;; The runtime table is filled once, from declarations lifted to module end.
;; A pin expands to the original expression and adds no evaluation.

(require (for-syntax racket/base
                     racket/file
                     racket/string))

(provide stage
         stage-ids
         stage-source-table
         register-stage-source!
         source-map-complete?
         repo-relative-path)

(define stage-ids
  '(bpe-merge
    input-embed
    block-ln1
    block-qkv
    kv-write
    attend
    block-proj-res
    block-ln2
    mlp-gelu
    block-mlp-res
    final-ln
    tied-logits
    sample))

(define stage-source-table (make-hasheq))

(define (register-stage-source! id info)
  (unless (hash? info)
    (raise-argument-error 'register-stage-source! "hash?" info))
  (hash-set! stage-source-table id info))

(define (source-map-complete?)
  (andmap (lambda (id) (hash-has-key? stage-source-table id)) stage-ids))

;; Absolute paths from the expander become "src/block.rkt". A path that
;; is already repo-relative is kept. Anything else is refused: a VM path
;; must not leak into the exported map.
(define (repo-relative-path src)
  (define s
    (cond
      [(path? src) (path->string src)]
      [(string? src) src]
      [else (format "~a" src)]))
  (cond
    [(regexp-match? #rx"^(src|tools)/[^/]+\\.rkt$" s) s]
    [else
     (define m (regexp-match #rx"(?:^|/)((?:src|tools)/[^/]+\\.rkt)$" s))
     (if m
         (cadr m)
         (error 'repo-relative-path "source is not a repo file: ~a" s))]))

(begin-for-syntax
  ;; syntax-position and syntax-span count characters, not bytes.
  ;; model.rkt's header contains multibyte code points, so a byte
  ;; offset would land before the form and clip the snippet.
  (define (source-text path)
    (file->string
     (cond
       [(path? path) path]
       [(string? path) path]
       [else (string->path (format "~a" path))])))

  (define (newline-count s)
    (for/sum ([c (in-string s)])
      (if (char=? c #\newline) 1 0)))

  (define (first-line s)
    (define m (regexp-match #rx"^[^\n]*" s))
    (string-trim (if m (car m) s)))

  (define (pin-info id-stx fn-stx expr-stx)
    (define src (syntax-source expr-stx))
    (define line (syntax-line expr-stx))
    (define pos (syntax-position expr-stx))
    (define span (syntax-span expr-stx))
    (unless (and src line pos span)
      (raise-syntax-error 'stage "missing source location" expr-stx))
    (define file (let ([s (if (path? src) (path->string src) (format "~a" src))])
                   (cond
                     [(regexp-match? #rx"^(src|tools)/[^/]+\\.rkt$" s) s]
                     [else
                      (define m (regexp-match #rx"(?:^|/)((?:src|tools)/[^/]+\\.rkt)$" s))
                      (unless m
                        (raise-syntax-error 'stage (format "source is not a repo file: ~a" s) expr-stx))
                      (cadr m)])))
    (define raw (source-text src))
    (define start (sub1 pos))
    (unless (and (>= start 0) (<= (+ start span) (string-length raw)))
      (raise-syntax-error 'stage "source span is outside the file" expr-stx))
    (define excerpt (substring raw start (+ start span)))
    (define end-line (+ line (newline-count excerpt)))
    ;; The closing delimiter sits on end-line. A span that ends on a
    ;; newline would count one past the last code line; the form's last
    ;; character is the closer, so the count of interior newlines is right.
    (define snippet (first-line excerpt))
    (when (string=? snippet "")
      (raise-syntax-error 'stage "empty source snippet" expr-stx))
    (hasheq 'file file
            'line line
            'endLine end-line
            'fn (symbol->string (syntax-e fn-stx))
            'snippet snippet)))

(define-syntax (stage stx)
  (syntax-case stx ()
    [(_ id fn expr)
     (let ([info (pin-info #'id #'fn #'expr)])
       (unless (memq (syntax-e #'id) '(bpe-merge input-embed block-ln1 block-qkv
                                                 kv-write attend block-proj-res
                                                 block-ln2 mlp-gelu block-mlp-res
                                                 final-ln tied-logits sample))
         (raise-syntax-error 'stage "unknown stage" #'id))
       (syntax-local-lift-module-end-declaration
        #`(register-stage-source! 'id #,(datum->syntax #'id info)))
       #'expr)]))
