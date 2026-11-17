#lang racket/base

;; Merge ranks from merges.txt. The first line of the GPT-2 file is
;; "#version: 0.2" and is not a merge. Any line starting with "#" is skipped.

(require racket/string
         "errors.rkt")

(provide read-merge-ranks)

(define (read-merge-ranks text)
  (unless (string? text)
    (raise-argument-error 'read-merge-ranks "string?" text))
  (define-values (pairs _rank)
    (for/fold ([pairs '()] [rank 0])
              ([line (in-list (regexp-split #rx"\r?\n" text))])
      (cond
        [(or (string=? line "") (string-prefix? line "#"))
         (values pairs rank)]
        [else
         (define parts (string-split line " "))
         (unless (= (length parts) 2)
           (raise-gpt2 (format "malformed merge line: ~a" line)))
         (values (cons (cons (cons (car parts) (cadr parts)) rank) pairs)
                 (add1 rank))])))
  (make-immutable-hash pairs))
