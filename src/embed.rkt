#lang racket/base

;; Token and position embedding lookup. wte is [vocab, d] and wpe is
;; [positions, d], both row-major. Positions are start .. start+n-1.
;; start + n cannot pass 1024.

(require racket/flonum
         "tensor.rkt")

(provide gpt2-context-length
         token-embed
         position-embed
         input-embed)

(define gpt2-context-length 1024)

(define (table-parts who t)
  (unless (tensor? t)
    (raise-argument-error who "tensor?" t))
  (define dims (tensor-dims t))
  (unless (= (length dims) 2)
    (raise-arguments-error who
                           "expected a rank-2 table"
                           "dims" (dims->string dims)))
  (values (car dims) (cadr dims) (tensor-data t)))

(define (copy-row! dest dest-row src src-row width)
  (define dest-base (* dest-row width))
  (define src-base (* src-row width))
  (let loop ([j 0])
    (unless (= j width)
      (flvector-set! dest (+ dest-base j) (flvector-ref src (+ src-base j)))
      (loop (add1 j)))))

(define (token-embed wte ids)
  (unless (list? ids)
    (raise-argument-error 'token-embed "list?" ids))
  (define-values (rows width data) (table-parts 'token-embed wte))
  (for ([id (in-list ids)])
    (unless (and (exact-nonnegative-integer? id) (< id rows))
      (raise-arguments-error 'token-embed
                             "token id out of range"
                             "id" id
                             "rows" rows)))
  (define n (length ids))
  (define out (make-flvector (* n width)))
  (for ([id (in-list ids)] [i (in-naturals)])
    (copy-row! out i data id width))
  (make-tensor (list n width) out))

(define (position-embed wpe n #:start [start 0])
  (unless (exact-nonnegative-integer? n)
    (raise-argument-error 'position-embed "exact-nonnegative-integer?" n))
  (unless (exact-nonnegative-integer? start)
    (raise-argument-error 'position-embed "exact-nonnegative-integer?" start))
  (define end (+ start n))
  (when (> end gpt2-context-length)
    (raise-arguments-error 'position-embed
                           "sequence is longer than the context"
                           "start" start
                           "length" n
                           "context" gpt2-context-length))
  (define-values (rows width data) (table-parts 'position-embed wpe))
  (when (> end rows)
    (raise-arguments-error 'position-embed
                           "sequence is longer than the position table"
                           "start" start
                           "length" n
                           "rows" rows))
  (define out (make-flvector (* n width)))
  (for ([i (in-range n)])
    (copy-row! out i data (+ start i) width))
  (make-tensor (list n width) out))

(define (input-embed wte wpe ids #:start [start 0])
  (tensor-add (token-embed wte ids)
              (position-embed wpe (length ids) #:start start)))
